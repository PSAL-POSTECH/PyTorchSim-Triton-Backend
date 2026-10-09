"""`tl.maximum` on an index is NOT AFFINE, so the address is an index table.

    tl.load(x_ptr + tl.maximum(t, 4))     4 on the first four lanes, t after

FOUR LINES, no mask and no loop. `max(t, 4)` cannot be an offset and a stride,
so the honest readings are two: the clamp NEVER FIRES and is the identity, or
it is real and every element gets its own address. This kernel is the second.

WHAT IT USED TO DO, and why the number below is the whole point.
`visitOperandMinSI`/`visitOperandMaxSI` picked one operand's state and returned
success -- the clamp went nowhere, the access came out a plain `tts.make_tptr`
with offsets [0] and strides [1], and the first four elements read x[0..3]
instead of x[4]. A WRONG ADDRESS, silently, in four lines. The baseline
recorded it as a 2.00107 numerical error carried under
`extra["upstream_miscompile"]` until 2026-09-06.

    ttir       1 arith.maxsi
    linalg     1 arith.maxsi  ->  tts.make_gather_scatter_tptr, 0 make_tptr on x

WHERE IT COMES FROM. A clamp on an index is what bounds-safe kernels write:
`tl.minimum(idx, N-1)`, `tl.maximum(idx, 0)`, the pair around a gather in
grid_sample and bilinear upsampling. upsample.py has four of them and they
survive only because a float round trip (`fptosi`) breaks the pattern before
PtrAnalysis sees it -- which is luck, not a guard.

WHAT WOULD BREAK IT AGAIN. Proving "the clamp never fires" from a bound this
cannot read: the proof needs the walk's range AND the bound as compile-time
constants, and a runtime bound is not one however obvious it looks to a reader
(ptr_minmax_scalar_operand.py is four loads whose bounds are `pid + 4096` and
`pid - 1`, and they take the index table for exactly this reason).

THIS IS UPSTREAM'S (rule 11): `max` is not affine for anyone using this IR, so
the fix is in PtrAnalysis and not in a pass of ours.

A clamp that fires cannot be a stride. Four lines, one gather, and the number is 0 only when the clamp reaches the address.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_maximum_clamp_is_not_affine_kernel(x_ptr, o_ptr, N: tl.constexpr):
        t = tl.arange(0, N)
        tl.store(o_ptr + t, tl.load(x_ptr + tl.maximum(t, 4)))

    return ptr_maximum_clamp_is_not_affine_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"][torch.maximum(torch.arange(N), torch.tensor(4))]}


SPEC = KernelSpec(
    name="ptr_maximum_clamp_is_not_affine_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE NUMBER ALONE CANNOT SAY IT: a make_tptr over `t` answers 2.00107 and a
    # gather over `max(t, 4)` answers 0, but only the op counts say WHICH shape
    # the address took, and the clamp surviving into 07-regularized is the claim.
    expect_ir={"01-ttir.mlir": {"arith.maxsi": 1},
               "07-regularized.mlir": {"arith.maxsi": 1,
                                    "tts.make_gather_scatter_tptr": 1}},
)
