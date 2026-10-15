"""x + 1 over 384x256 bf16: 128 rows per program. Blocked as XLA blocks it (validation/vector/emit_kernels.py)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, BM, BC = 384, 256, 128, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def add1_kernel(x_ptr, o_ptr):
        rows = tl.program_id(0) * 128 + tl.arange(0, 128)
        cols = tl.arange(0, 256)
        idx = rows[:, None] * 256 + cols[None, :]
        x = tl.load(x_ptr + idx)
        tl.store(o_ptr + idx, x + 1.0)

    return add1_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(R * C, generator=g, dtype=torch.float32).to(torch.bfloat16)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, C)
    return {"o": (x + 1).reshape(-1)}


SPEC = KernelSpec(
    name="add1_kernel",
    kernel=kernel,
    signature={"x_ptr": "*bf16", "o_ptr": "*bf16"},
    constexprs={},
    args=[Arg("x", IN, "bfloat16", (R * C,)), Arg("o", OUT, "bfloat16", (R * C,))],
    grid=(3,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
)
