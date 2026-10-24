"""bf16 remainder of two [16] rows: arith.remf has no RVV instruction and llc would make each element
a fmodf call nothing links. The harness takes any stop as this refusal, so the name (p08's
RVV_NO_INSTRUCTION) is pinned by math_late_cases; on the tree before it ld failed unnamed."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop
BM = 16
def kernel():
    import triton
    import triton.language as tl
    @triton.jit
    def frem_bf16_refused_kernel(x_ptr, y_ptr, o_ptr, BM: tl.constexpr):
        r = tl.arange(0, BM)
        x = tl.load(x_ptr + r)
        y = tl.load(y_ptr + r)
        tl.store(o_ptr + r, x % y)
    return frem_bf16_refused_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": (torch.randn(BM, generator=g) * 10).to(torch.bfloat16), "y": (torch.rand(BM, generator=g) + 1.0).to(torch.bfloat16)}
def reference(i):
    import torch
    x, y = i["x"], i["y"]
    return {"o": torch.remainder(x.float(), y.float()).to(torch.bfloat16)}
SPEC = KernelSpec(name="frem_bf16_refused_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "o_ptr": "*bf16", "BM": "constexpr"},
    constexprs={"BM": BM},
    args=[Arg("x", IN, "bfloat16", (BM,)), Arg("y", IN, "bfloat16", (BM,)), Arg("o", OUT, "bfloat16", (BM,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-2, atol=1e-2, expect="refuse",
    expect_stop=Stop("lower", Reason.MATH_UNLOWERED),
    expect_why="pytorchsim-refuse-unlowered-math: arith.remf has no RVV instruction")
