"""t2u-cat-multiple-bases-fail -- tl.cat of two pointer tensors.

TritonToUnstructuredPass.cpp:461. The only Case in the worklist TypeSwitch that
emits its own diagnostic before failing:

    error: Do not support gather / scatter with multiple bases yet

followed by the module warning at :577. tl.cat is the one Triton op that can put
two DIFFERENT base pointers into one tensor, which is exactly what the
base+offset representation cannot express.

torch equivalent: torch.cat([x, y]).

Coverage: the pass must BAIL. Not runnable end to end.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fail_cat_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        ps = tl.cat(x_ptr + offs, y_ptr + offs, can_reorder=True)
        tl.store(o_ptr + tl.arange(0, 2 * BLOCK), tl.load(ps))

    return fail_cat_kernel


def make_inputs(torch, seed=0):
    return {"x": torch.zeros(N, dtype=torch.float32),
            "y": torch.zeros(N, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.zeros(2 * N, dtype=torch.float32)}


SPEC = KernelSpec(
    name="fail_cat_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (2 * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse",
    expect_why="OUR convert_unsupported_pointers stops it first: `a tile of "
               "pointers passes through tensor.insert_slice`. tt.cat being the "
               "Case that TypeSwitch misses is upstream of that and no longer "
               "the observable (census 2026-09-04)",
)
