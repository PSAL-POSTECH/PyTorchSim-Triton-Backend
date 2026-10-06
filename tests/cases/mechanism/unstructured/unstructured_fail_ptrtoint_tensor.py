"""t2u-ptrtoint-tensor-fail -- tt.ptr_to_int on a TENSOR of pointers.

TritonToUnstructuredPass.cpp:298. The pass can only track one base pointer per
chain, so a tensor-of-pointers -> tensor-of-ints conversion is refused outright:
`return failure()` with no diagnostic of its own, which then trips the
module-level warning at :577 (t2u-pass-failure-warning-only).

Own file because processUnstructuredPtrs returns on the FIRST failure -- any
second failing construct in the same kernel would never be reached.

    triton-shared-opt --triton-to-unstructured
    warning: Cannot transform tensor of pointers into a single base pointer
             with tensor of offsets

UPSTREAM STILL BAILS AND THAT IS NO LONGER THE END. convert_unsupported_pointers takes
the tensor of pointers the warning leaves behind down to i64, so this kernel now
compiles and RUNS. What it stored -- raw addresses -- has no torch answer, and
the reference was a stub of zeros that only made sense while nothing ran; the
run reported 128/128 wrong against it, which was the stub failing, not the
addresses.

So it stores the DIFFERENCE from the base instead. That is the same
`tt.ptr_to_int` on the same tensor of pointers, still refused by the same
upstream branch, and its answer is `i * sizeof(f32)` no matter where the buffer
lands -- checkable, and it pins that the dissolved addresses come out
CONSECUTIVE rather than merely non-zero.

Coverage: the pass must BAIL -- and the tensor of pointers it leaves behind is dissolved below it, so this runs. The answer is the byte distance from the base, which is the same wherever the buffer is.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fail_ptrtoint_tensor_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        ps = x_ptr + offs
        # tt.ptr_to_int on the TENSOR (what upstream bails on), minus the same
        # on the SCALAR base, so the answer does not depend on the address.
        tl.store(o_ptr + offs, ps.to(tl.uint64) - x_ptr.to(tl.uint64))

    return fail_ptrtoint_tensor_kernel


def make_inputs(torch, seed=0):
    return {"x": torch.zeros(N, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.arange(N, dtype=torch.int64) * 4}


SPEC = KernelSpec(
    name="fail_ptrtoint_tensor_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*i64", "BLOCK": "constexpr"},
    constexprs={"BLOCK": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "int64", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
