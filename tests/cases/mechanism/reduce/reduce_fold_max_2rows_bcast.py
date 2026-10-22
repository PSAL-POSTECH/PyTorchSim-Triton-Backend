"""A max folded over 2 rows on 64 lanes, subtracted back from every row: the UNIFORM [1,64] result
is copied in-lane into every row. Breaks if the in-lane copy takes another row; the values are
positive, so a fold that stops masking the other lanes still passes here."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 2, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_fold_max_2rows_bcast_kernel(x_ptr, o_ptr, R: tl.constexpr, C: tl.constexpr):
        rr = tl.arange(0, R)[:, None]
        cc = tl.arange(0, C)[None, :]
        x = tl.load(x_ptr + rr * C + cc)
        y = tl.exp(x)
        tl.store(o_ptr + rr * C + cc, y - tl.max(y, axis=0)[None, :])
    return reduce_fold_max_2rows_bcast_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.rand(R, C, generator=g) + 1.0}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": (torch.exp(x) - torch.exp(x).amax(0, keepdim=True))}


SPEC = KernelSpec(
    name="reduce_fold_max_2rows_bcast_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    expect="pass",
)
