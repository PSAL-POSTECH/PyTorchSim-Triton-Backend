"""W8A8 -- an int32 contraction, then a per-output-channel dequant epilogue.

    acc = xq @ wq                      int8 values, INTEGER accumulation
    out = acc * sx[:, None] * sw[None, :] + bias

sx is one scale per TOKEN (the A8 half), sw is one scale per OUTPUT CHANNEL (the
W8 half). This is SmoothQuant / LLM.int8()'s shape and what every W8A8 linear in
vLLM and TensorRT-LLM computes.

HOW IT DIFFERS FROM quant_int4_matmul.py, WHICH IS THE OTHER QUANTISED MATMUL
HERE, AND THE DIFFERENCES ARE NOT COSMETIC.

  1. THE SCALE IS ON THE OTHER SIDE OF THE SUM. int4's scale is per GROUP along
     K, so it depends on the contracted index and CANNOT leave the reduction --
     it has to be applied to the weight operand, element by element, before the
     dot. A per-output-channel scale depends only on n, so it factors straight
     out of the K sum and the dequant becomes an EPILOGUE on the [M, N] result.
     Same word, opposite position in the dataflow: one rewrites an operand, the
     other rescales an output. It is also why W8A8 is the cheap one -- K
     multiplies per output become one.

  2. THERE IS NO UNPACK. One value per container means no index-dependent shift,
     no nibble mask, no bias by 8 -- the whole first third of the int4 kernel is
     gone. What replaces it is:

  3. THE ACCUMULATOR IS AN INTEGER. int4 converts to fp32 and hands `tl.dot` a
     float problem; the quantisation never reaches the arithmetic. Here the
     contraction itself is int32, which is the point of W8A8 -- on a GPU it is
     `mma.s8.s8.s32`, four times the fp16 throughput, and the fp32 rescale is
     paid once per output instead of once per MAC.

     AND THAT IS THE QUESTION THIS FILE ASKS. Rulebook fact 5: the systolic array
     is float throughout -- serialisers, weights, accumulator -- and there is no
     integer MAC to lower to. So this contraction cannot become a `tt.dot` at
     all. It is written as a loop over N with `tl.sum` on an int32 product, which
     is a VECTOR-UNIT integer multiply-accumulate: `vmul.vv` / `vredsum.vs`
     rather than the array. Whether that path exists and stays integer is exactly
     what stage 3 onwards has to answer, and no other kernel here asks it -- every
     other reduction in the suite is float.

WHY BOTH SCALES, AND WHY PER-TOKEN ON THE ACTIVATIONS. The reason activations are
hard to quantise is that their outliers are per-CHANNEL and enormous -- a handful
of feature dimensions run 100x the rest, and a single tensor scale spends its
whole range on them. SmoothQuant migrates that difficulty into the weights with an
offline per-channel smoothing factor, after which a plain per-token activation
scale is enough. So the two scales here are on different axes on purpose: sx
broadcasts along N, sw along M, and neither is a tensor scalar.

STORAGE IS i32 BECAUSE THIS ABI HAS NO i8. The VALUES are int8-ranged; the
containers are 32 bits. That costs nothing to the question being asked, which is
about the accumulator's type and not the operand's width. Headroom check: K * 128
* 127 is about 5.2e5 at K = 32, so the int32 accumulator cannot overflow and the
`acc` output is EXACT -- any mismatch in it is a lowering fault, not arithmetic,
which is why it is stored as well as `out`.

An integer contraction the systolic array cannot take, and a dequant that is an epilogue rather than an operand rewrite -- both the opposite of quant_int4_matmul.py. `acc` is exact by construction.

LAYOUT PIN. `expect_ir` holds `torchsim.copy` in 04-passes/04-torchsim-materialize-layouts.mlir at 0: the layouts come down decided, so materialize-layouts writes no copy of its own.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 16, 32, 16            # tokens, hidden, output channels


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def int8_dequant_matmul_kernel(xq_ptr, wq_ptr, sx_ptr, sw_ptr, b_ptr,
                                   acc_ptr, o_ptr, M: tl.constexpr,
                                   K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)

        xq = tl.load(xq_ptr + rm[:, None] * K + rk[None, :])      # [M, K] int32

        # INTEGER CONTRACTION, one output channel at a time so every tile stays
        # rank 2 and the fold stays on the inner axis. Not a tl.dot: the array
        # has no integer MAC, so this has to reach the vector unit instead.
        acc = tl.zeros((M, N), dtype=tl.int32)
        for j in tl.static_range(N):
            wj = tl.load(wq_ptr + rk * N + j)                     # [K] int32
            aj = tl.sum(xq * wj[None, :], axis=1)                 # [M] int32
            acc = tl.where(rn[None, :] == j, aj[:, None], acc)

        off = rm[:, None] * N + rn[None, :]
        tl.store(acc_ptr + off, acc)

        # DEQUANT AS AN EPILOGUE. Neither scale could be here if either depended
        # on k; both do not, which is what per-token and per-channel mean.
        sx = tl.load(sx_ptr + rm)[:, None]
        sw = tl.load(sw_ptr + rn)[None, :]
        tl.store(o_ptr + off,
                 acc.to(tl.float32) * sx * sw + tl.load(b_ptr + rn)[None, :])

    return int8_dequant_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # int8-ranged values in int32 containers -- symmetric, so no zero point.
    return {"xq": torch.randint(-128, 128, (M * K,), generator=g,
                                dtype=torch.int32),
            "wq": torch.randint(-128, 128, (K * N,), generator=g,
                                dtype=torch.int32),
            "sx": torch.rand(M, generator=g, dtype=torch.float32) * 0.02 + 0.005,
            "sw": torch.rand(N, generator=g, dtype=torch.float32) * 0.02 + 0.005,
            "b": torch.randn(N, generator=g, dtype=torch.float32) * 0.1}


def reference(inputs):
    import torch
    # float64 for the integer matmul: every value here is exactly representable,
    # so this is the int32 result and not an approximation of it.
    xq = inputs["xq"].reshape(M, K).to(torch.float64)
    wq = inputs["wq"].reshape(K, N).to(torch.float64)
    acc = (xq @ wq).to(torch.int32)
    out = (acc.to(torch.float32) * inputs["sx"][:, None] * inputs["sw"][None, :]
           + inputs["b"][None, :])
    return {"acc": acc.reshape(-1).contiguous(),
            "o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="int8_dequant_matmul_kernel", kernel=kernel,
    signature={"xq_ptr": "*i32", "wq_ptr": "*i32", "sx_ptr": "*fp32",
               "sw_ptr": "*fp32", "b_ptr": "*fp32", "acc_ptr": "*i32",
               "o_ptr": "*fp32", "M": "constexpr", "K": "constexpr",
               "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("xq", IN, "int32", (M * K,)),
          Arg("wq", IN, "int32", (K * N,)),
          Arg("sx", IN, "float32", (M,)),
          Arg("sw", IN, "float32", (N,)),
          Arg("b", IN, "float32", (N,)),
          Arg("acc", OUT, "int32", (M * N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"04-passes/04-torchsim-materialize-layouts.mlir": {"torchsim.copy": 0}},
)
