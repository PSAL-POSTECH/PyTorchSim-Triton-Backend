"""DeepSeek-V3 block quantisation -- one scale per (128, 128) TILE of the weight,
so the scale depends on the contracted index and has to be re-read every K step.

    for kb in blocks of K:
        acc += (xq[:, kb] @ wq[kb, :]) * sx[:, kb] * sw[kb, nb]
    out = acc

sw is indexed by BOTH block coordinates. sx is per token per K-block (DeepSeek's
1 x 128 activation group). Neither can leave the loop.

WHY BLOCKS AND NOT PER-CHANNEL
  Per-output-channel scaling (int8_dequant_matmul.py) assumes the error inside a
  column is uniform along K. At 8 bits that is close enough; at the 4-6 bit
  effective width the big MoE checkpoints run at, it is not -- a single row of K
  with an outlier drags the whole column's scale and everything else in the
  column quantises to a handful of levels. Blocking along BOTH axes caps how far
  an outlier's influence spreads: 128x128 elements, and the next tile over is
  unaffected. DeepSeek-V3 ships its weights this way and its activations as
  1 x 128 groups, and that pairing is what makes its fp8 training stable.

THE STRUCTURAL POINT, AND IT IS THE OPPOSITE OF BOTH EXISTING QUANT KERNELS
  int8_dequant_matmul.py's scales do not depend on k, so the dequant factors
  clean out of the sum and becomes an epilogue on [M, N] -- ONE rescale per
  output. quant_int4_matmul.py's scale does depend on k, but only on k, so it is
  folded into the weight operand ONCE, before a single full-K dot.

  Here the scale depends on k AND n. That kills both escapes:
    * it cannot be an epilogue, because different k-blocks contributing to the
      same output element carry different scales;
    * it cannot be a one-time operand rewrite either, because the K loop is the
      thing that indexes it -- the load is INSIDE the loop and its address moves
      with the induction variable.
  So the contraction becomes a sequence of PARTIAL products, each dequantised on
  its own before being added into a float accumulator. The accumulator is float
  and the partials are integer, which is exactly the split real fp8 hardware
  uses (tensor-core partial in low precision, promotion to fp32 every 128 steps).
  What this asks the backend for is a loop-carried [M, N] accumulator whose
  update is a dot followed by two rank-1 broadcasts -- one along N, one along M
  -- with a fresh scale load each trip.

INT8 SURROGATE. The real thing is e4m3 weights with an fp32 block scale. fp8 is
gated in this backend's frontend (see ttir.py), so the operands here are
int8-ranged values in int32 containers, promoted to fp32 for the dot. That
changes nothing about the question, which is entirely about WHERE the scale is
read and what it forces the loop to carry; the dtype of the mantissa is not
part of it. Values stay under 2^24 so the promotion and the partial sums are
exact and only the final float adds carry any error.

BLOCK SIZE 16, not 128, so the loop is two trips and the IR stays readable. The
K-block and the scale block are deliberately the same size (BK == BS), which is
what DeepSeek does; if they differed the scale index would be a second division.

GREEN SINCE 2026-09-04. vectorize refused this with "operands of one
elementwise op are banked on different iteration dims": one buffer a matmul pins
on one axis sits in an all-parallel op beside buffers that follow the op onto the
other. select_lane_axis owned that decision and did not know it -- its
`_is_one_iteration_dim` reported only the disagreements a widen could already
fix, so the ladder never reached step 4 and the refusal fell out a pass later.

The pinned operand here is the op's `outs`, and a relayout mints a new value
while the op already produces that one. So the cut goes on the CONSUMER instead:
the matmul that pins it reads a copy banked its way, and the producing op is free
to follow its own operands. Same trip, no new rewrite. Costs 2 round trips.

A scale indexed by BOTH block axes, so it is loaded inside the K loop -- neither an epilogue (int8_dequant_matmul) nor a one-time operand rewrite (quant_int4_matmul). Float accumulator, int partials.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 16, 32, 32          # tokens, hidden, output channels
BS = 16                       # quantisation block, both axes (== the K step)
KB, NB = K // BS, N // BS     # 2 x 2 grid of weight scales


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def block_scale_2d_matmul_kernel(xq_ptr, wq_ptr, sx_ptr, sw_ptr, o_ptr,
                                     M: tl.constexpr, K: tl.constexpr,
                                     N: tl.constexpr, BS: tl.constexpr,
                                     KB: tl.constexpr, NB: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rb = tl.arange(0, BS)

        acc = tl.zeros((M, N), dtype=tl.float32)
        for kb in tl.static_range(KB):
            rk = kb * BS + rb                                   # this K block

            xq = tl.load(xq_ptr + rm[:, None] * K + rk[None, :])   # [M, BS]
            wq = tl.load(wq_ptr + rk[:, None] * N + rn[None, :])   # [BS, N]

            # THE TWO SCALE LOADS, BOTH INSIDE THE LOOP. sw's address moves with
            # kb along one axis and with rn // BS along the other -- a 2-D block
            # index, which is why it cannot be hoisted or folded into an operand.
            sw = tl.load(sw_ptr + kb * NB + (rn // BS))            # [N]
            sx = tl.load(sx_ptr + rm * KB + kb)                    # [M]

            part = tl.dot(xq.to(tl.float32), wq.to(tl.float32))    # [M, N]
            acc += part * sx[:, None] * sw[None, :]

        tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc)

    return block_scale_2d_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # int8-ranged values in int32 containers; the scales are what a real
    # absmax/127 would look like for activations of order 1.
    return {"xq": torch.randint(-128, 128, (M, K), generator=g,
                                dtype=torch.int32),
            "wq": torch.randint(-128, 128, (K, N), generator=g,
                                dtype=torch.int32),
            "sx": torch.rand(M, KB, generator=g, dtype=torch.float32) * 0.02
                  + 0.005,
            "sw": torch.rand(KB, NB, generator=g, dtype=torch.float32) * 0.02
                  + 0.005}


def reference(inputs):
    import torch
    xq = inputs["xq"].to(torch.float64)
    wq = inputs["wq"].to(torch.float64)
    sx = inputs["sx"].to(torch.float64)
    sw = inputs["sw"].to(torch.float64)
    acc = torch.zeros(M, N, dtype=torch.float64)
    for kb in range(KB):
        sl = slice(kb * BS, (kb + 1) * BS)
        part = xq[:, sl] @ wq[sl, :]
        acc += part * sx[:, kb:kb + 1] * sw[kb].repeat_interleave(BS)[None, :]
    return {"o": acc.to(torch.float32)}


SPEC = KernelSpec(
    name="block_scale_2d_matmul_kernel", kernel=kernel,
    signature={"xq_ptr": "*i32", "wq_ptr": "*i32", "sx_ptr": "*fp32",
               "sw_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "K": "constexpr", "N": "constexpr", "BS": "constexpr",
               "KB": "constexpr", "NB": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "BS": BS, "KB": KB, "NB": NB},
    args=[Arg("xq", IN, "int32", (M, K)),
          Arg("wq", IN, "int32", (K, N)),
          Arg("sx", IN, "float32", (M, KB)),
          Arg("sw", IN, "float32", (KB, NB)),
          Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
