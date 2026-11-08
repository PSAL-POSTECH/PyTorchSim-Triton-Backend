"""o is a view of x's storage from element 1280 on (placement: x at 0, o 5120 bytes past it), and the kernel
writes o's first 256 elements alone: the rest of o is x's, as a view of one storage reads. Breaks if the
wrapper ignores the placement file or gives overlapping arguments disjoint buffers (o's tail then reads 0)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, AT, W = 2048, 1280, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def launch_args_overlapping_in_one_storage_kernel(x_ptr, o_ptr, W: tl.constexpr):
        r = tl.arange(0, W)
        tl.store(o_ptr + r, tl.load(x_ptr + r) * 2.0)

    return launch_args_overlapping_in_one_storage_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": torch.cat([x[:W] * 2.0, x[AT + W:]])}


SPEC = KernelSpec(
    name="launch_args_overlapping_in_one_storage_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "W": "constexpr"},
    constexprs={"W": W},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N - AT,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-6, atol=1e-6,
    extra={"placement": {"x": 0, "o": AT * 4}},
)
