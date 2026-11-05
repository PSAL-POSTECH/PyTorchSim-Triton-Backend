"""A store masked from below, `y_ptr + ((-32) + r)` with `r >= 32`: the TOG DMA writes y[0..95], from the
clamp's low corner. Breaks if the TOG sizes the store to the mask but keeps the unmasked start -32
(tog.cpp names -32, togcheck finds y written from -32)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, LO = 128, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_tog_store_starts_at_the_clamp_low(x_ptr, y_ptr, N: tl.constexpr, LO: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(y_ptr + ((-LO) + r), tl.load(x_ptr + r) * 2.0, mask=r >= LO)

    return mask_tog_store_starts_at_the_clamp_low


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g)}


def reference(inputs):
    return {"y": inputs["x"][LO:] * 2.0}


SPEC = KernelSpec(
    name="mask_tog_store_starts_at_the_clamp_low", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr", "LO": "constexpr"},
    constexprs={"N": N, "LO": LO},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", OUT, "float32", (N - LO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"tog.cpp": {"-32": 0}}, expect_stage="togcheck",
)
