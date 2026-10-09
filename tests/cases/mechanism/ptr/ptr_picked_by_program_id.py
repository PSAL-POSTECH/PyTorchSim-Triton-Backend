"""Pointers picked by program id, then one body: Inductor's shared-body foreach kernel.

WHAT IT IS. torch._foreach_* over a parameter list (Adam's step) becomes one kernel whose
sub-kernels share a body; the branch on `pid` only picks each sub-kernel's arguments:

    if pid < 1: a = x_ptr; o = out0_ptr; s = alpha0
    else:       a = y_ptr; o = out1_ptr; s = alpha1
    tl.store(o + idx, tl.load(a + idx) * s)

In TTIR that is an scf.if yielding pointers (and arith.selects, once the canonicalizer has
folded a branch that only picks), and the body's make_tptr base is the if's result.

WHAT IT USED TO DO. convert-accesses-to-transfer names a transfer's DRAM argument statically,
so a base that is no function argument was refused:

    _Fatal: make_tptr base does not trace to a function arg: scf.if ... -> (!ptr.ptr ...)

The pass triton-torchsim-sink-into-pointer-ifs now moves the body into each branch (one copy
per sub-kernel, exactly one of which runs), so every base is an argument again. The float
picked beside the pointers rides along as a runtime scalar.

torch equivalent: out0 = x * alpha0; out1 = y * alpha1
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
ALPHA0, ALPHA1 = 0.5, -2.0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_picked_by_program_id(x_ptr, y_ptr, out0_ptr, out1_ptr, alpha0, alpha1,
                                 N: tl.constexpr):
        pid = tl.program_id(0)
        if pid < 1:
            a = x_ptr
            o = out0_ptr
            s = alpha0
        else:
            a = y_ptr
            o = out1_ptr
            s = alpha1
        idx = tl.arange(0, N)
        tl.store(o + idx, tl.load(a + idx) * s)
    return ptr_picked_by_program_id


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "y": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"out0": (inputs["x"] * ALPHA0).contiguous(),
            "out1": (inputs["y"] * ALPHA1).contiguous()}


SPEC = KernelSpec(
    name="ptr_picked_by_program_id", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out0_ptr": "*fp32", "out1_ptr": "*fp32",
               "alpha0": "fp32", "alpha1": "fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)),
          Arg("y", IN, "float32", (N,)),
          Arg("out0", OUT, "float32", (N,)),
          Arg("out1", OUT, "float32", (N,))],
    grid=(2,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"alpha0": ALPHA0, "alpha1": ALPHA1}},
    rtol=1e-4, atol=1e-4,
)
