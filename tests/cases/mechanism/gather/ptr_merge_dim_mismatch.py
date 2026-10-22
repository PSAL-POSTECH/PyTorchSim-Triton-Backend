"""An address whose two halves name different gather axes: `minimum(r % M, j[c])`.

The address is

    tl.minimum(r8[:, None] % M, j8[None, :])

a modulo of an arange on dim 0 against a loaded index on dim 1. The structured
pass and the unstructured re-analysis used to pick different axes for it, and
mergeUnstructuredState refused the merge: the load fell to a per-element
tts.gather.

SINCE MULTI-AXIS GATHERS, the minimum varies on both axes and is one index over
both (tts.make_gather_scatter_tptr, gather_scatter_dims = [0, 1]): the two
passes agree that both axes gather, and the merge goes through. Remove the `% M`
or the loaded `j8` and the address varies on one axis only.

8x8 so the two axes are easy to tell apart in the IR; M = 64 makes `r8 % 64` the
identity on 0..7, keeping a modulo state on dim 0 without arithmetic to reason
about. idx is zeroed, so the result is in[0] over the whole tile.

The structured and unstructured passes see different gather axes; the address is one index over both.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
M = 64
R = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_merge_dim_mismatch_kernel(in_ptr, idx_ptr, o_ptr,
                                      M: tl.constexpr, R: tl.constexpr):
        r8 = tl.arange(0, R)

        j8 = tl.load(idx_ptr + r8)
        s = tl.load(in_ptr + tl.minimum((r8[:, None] % M), j8[None, :]))
        tl.store(o_ptr + r8[:, None] * R + r8[None, :], s)

    return ptr_merge_dim_mismatch_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": torch.zeros(R, dtype=torch.int32)}


def reference(inputs):
    """in[minimum(i % 64, idx[j])] for i, j in 0..7: in[0] over the tile, idx being zero."""
    import torch
    return {"o": inputs["in"][0].expand(R * R).clone()}


SPEC = KernelSpec(
    name="ptr_merge_dim_mismatch_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32",
               "M": "constexpr", "R": "constexpr"},
    constexprs={"M": M, "R": R},
    args=[Arg("in", IN, "float32", (N,)),
          Arg("idx", IN, "int32", (R,)),
          Arg("o", OUT, "float32", (R, R))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
