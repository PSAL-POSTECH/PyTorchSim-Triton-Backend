"""A min folded over 2 rows on 64 lanes and stored: lanes 2-127 push +inf, not the 0 their banks
hold. Breaks if the fold stops masking the lanes its source does not ride."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 2, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_fold_min_2rows_exp_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
        rr = tl.arange(0, R)[:, None]
        cc = tl.arange(0, C)[None, :]
        x = tl.load(x_ptr + rr * C + cc)
        tl.store(o_ptr + tl.arange(0, C), tl.min(tl.exp(x) + 1.0, axis=0))
    return reduce_fold_min_2rows_exp_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.rand(R, C, generator=g) + 1.0}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": (torch.exp(x)+1).amin(0)}


SPEC = KernelSpec(
    name="reduce_fold_min_2rows_exp_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    expect="pass",
)
