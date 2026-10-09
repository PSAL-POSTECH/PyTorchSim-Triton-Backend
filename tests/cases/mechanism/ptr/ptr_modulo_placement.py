"""WHICH AXES a modulo covers, and what each answer costs.

A wrap is `(...) mod W` applied to some part of the address. Which part decides
everything below it:

    한 축만        shape[d] = W        그 축만 dram_stride 를 내놓는다
    합 전체        flat_shape = W      모든 축이 dram_stride 를 내놓는다

A size-1 axis holds one element, so a modulo of it is the identity and it is
never a candidate. One axis wider than 1 -> the wrap is that axis's. Two or more
-> the `%` is on their SUM, and `(a+b) mod M` is not `(a mod M)+(b mod M)`, so
that is a different fact with a slot of its own.

THREE LOADS, ONE OF EACH PLACEMENT. All three are structured; what differs is
how much of the descriptor survives:

  load 1  (rm*32 + rn) % 3            flat_shape.  dram_stride [0, 0]
  load 2  (r4*64 + r8*8 + r8) % 3     flat_shape.  dram_stride [0, 0, 0]
  load 3  (r4 % 3)[:,None,None]*64    shape[0].    dram_stride [0, 8, 1]
          + r8[None,:,None]*8 + r8                 <- two axes still WALK

THE MODULUS IS 3 AND NOT 8, and that is the file working. With 8 the first two
are not sum wraps at all: 8 divides 32, 64 and 8, so every term but the last is
DEAD inside the `%` and the whole-laps rule leaves those axes standing still,
leaving a wrap on one axis. A modulus that divides no coefficient is what makes the sum
the thing being wrapped.

AND IT IS WHAT MAKES THE NUMBER THE JUDGE. `expect_ir` counts TOKENS and a
placement is not one -- a flat wrap is an operand the printer shows only in
`operandSegmentSizes`, so the count below cannot tell it from a per-axis wrap
and did not: the file sat here claiming two flat wraps it had stopped
producing. Under a modulus that divides the coefficients the two placements
compute the SAME numbers, which is why nothing failed. Under 3 they do not:
`(rm*32 + rn) % 3` is not `rn % 3`, so a wrap placed on one axis answers wrong
and `reference()` says so. The pin for this file is its own arithmetic.

Load 3 is the point of the file: the `%` is on a rank-1 tensor and expand_dims
only adds size-1 axes, so nothing became ambiguous and only axis 0 pays.

WHAT IT USED TO PIN. All three were refused, for three reasons that were really
one gatekeeper each: a rank table in visitOperandRem, a `hasModulo() && rank > 2`
guard in visitOperandExpandDims, and no slot for a sum wrap. The refusals fell on
2026-08-31; the file's own sentence about placement ambiguity was the rule all
along.

torch equivalent: see reference().

One modulo of each placement -- on the sum twice, on a single axis once -- and the dram strides that survive each.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 3
BM = 16
BN = 32
D0, D1, D2 = 4, 8, 8
N = 1024
NO = BM * BN + D0 * D1 * D2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_modulo_placement_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                      BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        r4 = tl.arange(0, 4)
        r8 = tl.arange(0, 8)

        # 1 rank 2, neither dim is 1 -> which axis wraps?
        a = tl.load(x_ptr + ((rm[:, None] * BN + rn[None, :]) % M))

        # 2 three wide axes -> same ambiguity as 1, one rank higher
        idx3 = (r4[:, None, None] * 64 + r8[None, :, None] * 8
                + r8[None, None, :])
        b = tl.load(x_ptr + (idx3 % M))

        # 3 rank-1 modulo, then expanded with SIZE-1 axes -> still placeable
        c = tl.load(x_ptr + (r4 % M)[:, None, None] * 64
                    + r8[None, :, None] * 8 + r8[None, None, :])

        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], a)
        tl.store(o_ptr + BM * BN + r4[:, None, None] * 64
                 + r8[None, :, None] * 8 + r8[None, None, :], b + c)

    return ptr_modulo_placement_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    rm = torch.arange(BM).reshape(BM, 1)
    rn = torch.arange(BN).reshape(1, BN)
    a = x[((rm * BN + rn) % M).reshape(-1).long()]
    r4 = torch.arange(D0).reshape(D0, 1, 1)
    r8a = torch.arange(D1).reshape(1, D1, 1)
    r8b = torch.arange(D2).reshape(1, 1, D2)
    idx3 = r4 * 64 + r8a * 8 + r8b
    b = x[(idx3 % M).reshape(-1).long()]
    c = x[((r4 % M) * 64 + r8a * 8 + r8b).reshape(-1).long()]
    return {"o": torch.cat([a, b + c])}


SPEC = KernelSpec(
    name="ptr_modulo_placement_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"M": M, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (NO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE OBSERVABLE IS THE DESCRIPTOR, not a refusal: three make_tptr and no
    # gather, and load 3 keeps two dram strides while 1 and 2 keep none.
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.make_tptr": 5}},
)
