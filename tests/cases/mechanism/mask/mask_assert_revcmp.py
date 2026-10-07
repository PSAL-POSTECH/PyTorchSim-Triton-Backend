"""MaskAnalysis coverage -- NEGATIVE: cmp-scalar-lhs-tensor-rhs-assert.

THIS KERNEL IS EXPECTED TO FAIL, and the failure IS the observable:
triton-shared-opt ABORTS (SIGABRT, rc=134) inside --triton-to-structured on
  MaskAnalysis.cpp:555  assert(rhsState.scalar && "Unexpected case where rhs is
  not a scalar")
which is reached only when the scalar sits on the LEFT of the arith.cmpi -- here
`lo < rng`, which Triton emits as `arith.cmpi slt, splat(lo), rng`. No later
pass and no executable exist, so `reference()` below is never called.

Keep this kernel ALONE in its own file: the process dies, so nothing else in the
module would be analysed.

NEGATIVE kernel: aborts triton-shared-opt by design. Keep alone.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
LO = 0          # runtime value of `lo`; the mask is `lo < tl.arange(0, BLOCK)`


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_assert_revcmp_kernel(x_ptr, o_ptr, lo, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        rng = tl.arange(0, BLOCK)
        a = tl.load(x_ptr + offs, mask=lo < rng, other=0.0)   # splat on the LHS
        tl.store(o_ptr + offs, a)

    return mask_assert_revcmp_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    # Never evaluated -- triton-shared-opt aborts (see the module docstring).
    # o[i] = x[i] if LO < (i % BLOCK) else 0.0   (the store is UNMASKED, so the
    # `other=0.0` lanes are written out as zeros -- they are not left untouched).
    import torch
    x = inputs["x"]
    rng = torch.arange(N, dtype=torch.int32) % BLOCK
    return {"o": torch.where(rng > LO, x, torch.zeros_like(x))}


SPEC = KernelSpec(
    name="mask_assert_revcmp_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "lo": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(N // BLOCK,),
    reference=reference,
    make_inputs=make_inputs,
    # `lo` is a runtime scalar, not a memref Arg -- without this the generated C
    # driver (pytorchsim_triton_compiler/lowering/wrapper.py:_scalars) passes NOTHING for it and the trailing
    # grid/pid i32s slide into its slot. Unreachable here (the compiler aborts),
    # but the reference above is only well-defined once `lo` is pinned.
    extra={"scalar_args": [("lo", "int32_t")], "scalar_values": {"lo": LO}},
    expect="refuse",
    expect_why="a C++ assert in MaskAnalysis, which IS the observable",
)
