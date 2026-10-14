"""MXFP4 / NVFP4 -- the block scale is a POWER OF TWO, so dequantising is a SHIFT.

    w   = ((packed >> (4*(k%2))) & 0xF) - 8      # 4-bit mantissa, one per element
    eb  = exp[k // G, n]                          # E8M0: an exponent, biased
    deq = w << eb                                 # the entire dequant. no multiply
    o   = x @ (deq * 2^-BIAS)

WHAT AN E8M0 SCALE IS, AND WHY IT IS NOT AN ORDINARY SCALE
  The OCP Microscaling (MX) formats and NVIDIA's NVFP4 both give each block of 32
  elements a shared scale, and both restrict that scale to a bare exponent --
  E8M0 is eight bits of exponent and NO mantissa and no sign, i.e. the scale can
  only ever be 2^e. That is a deliberate constraint, not a storage saving: the
  scale then costs one byte per 32 elements AND, more to the point, applying it
  is exact. Multiplying a float by 2^e touches only the exponent field; there is
  no rounding, no denormal surprise, and dequant-then-requantise round-trips.
  AWQ/GPTQ's fp16 group scale (quant_int4_matmul.py) has neither property -- it
  is a general float, so the dequant is a real multiply and it rounds.

  GPT-OSS ships MXFP4 for its MoE expert weights; Blackwell's tensor cores
  consume NVFP4 natively with the E8M0/E4M3 scale handled in hardware.

THE UINT8 SURROGATE, AND WHY IT ASKS THE SAME QUESTION
  The real element type is E2M1 -- one sign bit, two exponent bits, one mantissa
  bit, sixteen values. fp4 and fp8 are gated in this backend's frontend
  (ttir.py explains where), so this file holds the 4-bit payload as a
  signed integer mantissa in a packed int32, and the power-of-two scale as a
  BIASED NON-NEGATIVE EXPONENT applied with `<<`.

  That substitution is faithful in the way that matters. On a float mantissa the
  E8M0 scale is an exponent ADD; on an integer mantissa it is a left SHIFT --
  the same operation on the same field, and in both cases exact. What reaches
  the backend is an elementwise shift whose amount is a LOADED TENSOR broadcast
  from a coarser axis, and that is the op under test. It is not an op any other
  kernel here produces: quant_int4_matmul.py's scale broadcast is an fmul,
  int8_dequant_matmul.py's is an fmul in the epilogue, and the only other shifts
  in the suite are by a compile-time-known nibble offset.

  The bias is undone once, at the very end, by multiplying by 2^-BIAS -- itself a
  power of two and therefore exact in fp32. So `deq` is an EXACT int32 tensor and
  is stored as an output: any mismatch in it is a lowering fault, not arithmetic.
  Only `o`, which carries the dot, can drift.

  A VARIABLE SHIFT AMOUNT is the part to watch. `w << eb` with `eb` a tensor is
  `arith.shli` on a vector, which a lane can do; a backend that only has an
  immediate-shift form has to scalarise it, and that shows up as a loop over the
  vector rather than one instruction.

GROUP SIZE 8 instead of 32 so the ratio is visible in a readable tile; the
arithmetic is identical. Exponents are stored biased by 3, i.e. real scales of
2^-3 .. 2^3, which is a realistic dynamic range for one block of a weight row.

MX/NVFP4 shape: a power-of-two block scale applied by a VARIABLE shift rather than an fmul. `deq` is exact by construction, so a mismatch there is lowering and only `o` can carry arithmetic drift.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N, G = 16, 32, 16, 8      # tokens, hidden, out channels, MX block along K
BIAS = 3                        # stored exponent is e + BIAS, so eb in [0, 6]
INV_BIAS = 1.0 / (1 << BIAS)    # 0.125 -- a power of two, so exact in fp32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mxfp4_shift_scale_matmul_kernel(x_ptr, wq_ptr, e_ptr, deq_ptr, o_ptr,
                                        M: tl.constexpr, K: tl.constexpr,
                                        N: tl.constexpr, G: tl.constexpr,
                                        INV_BIAS: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)

        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])            # [M, K]

        # The E2M1 payload, two per byte, as a signed 4-bit mantissa.
        byte = tl.load(wq_ptr + (rk[:, None] // 2) * N + rn[None, :])
        w = ((byte >> ((rk[:, None] % 2) * 4)) & 0xF) - 8              # [K, N]

        # THE E8M0 SCALE. One biased exponent per block of G along K, per output
        # channel, stretched back over the block -- and applied as a SHIFT whose
        # amount is this loaded tensor. No multiply anywhere in the dequant.
        eb = tl.load(e_ptr + (rk[:, None] // G) * N + rn[None, :])     # [K, N]
        deq = w << eb                                                  # exact
        tl.store(deq_ptr + rk[:, None] * N + rn[None, :], deq)

        # Undoing the exponent bias is one multiply by 2^-BIAS for the whole
        # tensor, and 2^-BIAS is exactly representable, so this rounds nothing.
        wf = deq.to(tl.float32) * INV_BIAS
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(x, wf))

    return mxfp4_shift_scale_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, K, generator=g, dtype=torch.float32),
            # Packed E2M1-width payloads: byte b holds elements 2b and 2b+1.
            "wq": torch.randint(0, 256, (K // 2, N), generator=g,
                                dtype=torch.int32),
            # Biased E8M0 exponents in [0, 6]; real scale is 2^(eb - 3).
            "e": torch.randint(0, 7, (K // G, N), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    byte = inputs["wq"]
    lo = (byte & 0xF) - 8
    hi = ((byte >> 4) & 0xF) - 8
    w = torch.stack([lo, hi], dim=1).reshape(K, N)
    eb = inputs["e"].repeat_interleave(G, dim=0)
    deq = w << eb                                   # int32, exact
    wf = deq.to(torch.float32) * INV_BIAS
    return {"deq": deq.contiguous(), "o": inputs["x"] @ wf}


SPEC = KernelSpec(
    name="mxfp4_shift_scale_matmul_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "wq_ptr": "*i32", "e_ptr": "*i32",
               "deq_ptr": "*i32", "o_ptr": "*fp32", "M": "constexpr",
               "K": "constexpr", "N": "constexpr", "G": "constexpr",
               "INV_BIAS": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "G": G, "INV_BIAS": INV_BIAS},
    args=[Arg("x", IN, "float32", (M, K)),
          Arg("wq", IN, "int32", (K // 2, N)),
          Arg("e", IN, "int32", (K // G, N)),
          Arg("deq", OUT, "int32", (K, N)),
          Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # Looser than the other quant kernels on purpose: a 2^6 spread of block
    # scales inside one dot means the summands differ by six binades, so `o`
    # carries real cancellation. `deq` is checked at full precision anyway.
    rtol=1e-3, atol=1e-3,
    expect="blocked",
    expect_why="matmul K=32 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
