"""min/max where ONE SIDE IS A GATHER -- the clamp goes INTO the index.

Split out of the old tests/cases/ptr_reject.py.  Four loads in one file, one
per arm: neither operand is a scalar and neither is a readable range, so all
four take `visitOperandClamp`'s third answer -- one address per element, built
over the CLAMPED value.

THE SETUP -- four SEPARATE index segments, on purpose
  A gather address has no stride, so what identifies the four arms is WHICH
  loaded index tensor reaches the gather offset.  Each arm gets its own
  segment of idx[] and therefore its own SSA value:

      idx[  0:128] = i           -> `a`, the LHS of the minimum
      idx[128:256] = i           -> `b`, the RHS of the minimum
      idx[256:384] = 1800 + i    -> `c`, the LHS of the maximum
      idx[384:512] = 1800 + i    -> `d`, the RHS of the maximum

  Feeding all four from one segment would make the four
  make_gather_scatter_tptr ops indistinguishable after CSE.  The structured
  operands keep unique multipliers 3/5/7/11, and the point of those is that
  they must appear in the INDEX and never as a stride.

WHAT PROVES IT.  Each `gather_scatter_offset:` is now the arith.minsi/maxsi
RESULT and not one of its operands: the offset tile is a linalg.generic
computing `minsi(a, r*3)`, and the clamp is part of the address.

WHAT IT USED TO PIN.  The tie used to be broken on STRUCTUREDNESS -- the side
nobody could reason about won and the other was thrown away, so the gather
offset was `a` alone and `r*3` appeared nowhere.  Same rung, WRONG INDEX.  The
clamps here are no-ops by construction, which is why the number never said so.

minsi/maxsi with one gather operand on either side: the clamp goes into the index and the address is a table. Four index segments keep the four arms apart in the IR.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 2048
BLOCK = 128
BIG = 1800


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_minmax_unstructured_operand_kernel(in_ptr, idx_ptr, o_ptr,
                                               BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        a = tl.load(idx_ptr + r)                # small values
        b = tl.load(idx_ptr + BLOCK + r)        # small values
        c = tl.load(idx_ptr + 2 * BLOCK + r)    # large values
        d = tl.load(idx_ptr + 3 * BLOCK + r)    # large values

        # min, GATHER ON THE LEFT -> state = lhs, `a` becomes the offset
        p = tl.load(in_ptr + tl.minimum(a, r * 3))
        # min, GATHER ON THE RIGHT -> state = rhs, `b` becomes the offset
        q = tl.load(in_ptr + tl.minimum(r * 5, b))
        # max, GATHER ON THE LEFT -> state = lhs, `c` becomes the offset
        s = tl.load(in_ptr + tl.maximum(c, r * 7))
        # max, GATHER ON THE RIGHT -> state = rhs, `d` becomes the offset
        t = tl.load(in_ptr + tl.maximum(r * 11, d))

        tl.store(o_ptr + r, p + q + s + t)

    return ptr_minmax_unstructured_operand_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(BLOCK)
    idx = torch.cat([i, i, i + BIG, i + BIG]).to(torch.int32)
    return {"in": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": idx}


def reference(inputs):
    import torch
    x = inputs["in"]
    idx = inputs["idx"].long()
    r = torch.arange(BLOCK)
    a, b = idx[0:BLOCK], idx[BLOCK:2 * BLOCK]
    c, d = idx[2 * BLOCK:3 * BLOCK], idx[3 * BLOCK:4 * BLOCK]
    # each clamp below is a no-op by construction, so this matches both the
    # source and the gather-only code the compiler actually emits
    return {"o": x[torch.minimum(a, r * 3)] + x[torch.minimum(r * 5, b)]
                 + x[torch.maximum(c, r * 7)] + x[torch.maximum(r * 11, d)]}


SPEC = KernelSpec(
    name="ptr_minmax_unstructured_operand_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("in", IN, "float32", (N,)),
          Arg("idx", IN, "int32", (4 * BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,),
    # THE NUMBER CANNOT SAY IT: every clamp here is a no-op, so an index built
    # from one operand answers the same. What the count pins is that all four
    # addresses are index tables at all.
    expect_ir={"03-linalg.mlir": {"tts.make_gather_scatter_tptr": 4}},
    reference=reference,
    make_inputs=make_inputs,
)
