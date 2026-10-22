"""A max folded over 2 rows, stored (MVOUT) and subtracted from every row: the [1,64] result is
UNIFORM, so every lane copies it from its own bank, with no move between lanes. Breaks if the fold
stops masking the lanes its source does not ride, or the in-lane copy takes another row."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C = 2, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_fold_bcast_and_store_kernel(x_ptr, o_ptr, m_ptr, R: tl.constexpr, C: tl.constexpr):
        rr = tl.arange(0, R)[:, None]
        cc = tl.arange(0, C)[None, :]
        y = tl.exp(tl.load(x_ptr + rr * C + cc))
        m = tl.max(y, axis=0)
        tl.store(m_ptr + tl.arange(0, C), m)
        tl.store(o_ptr + rr * C + cc, y - m[None, :])
    return reduce_fold_bcast_and_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": -(torch.rand(R, C, generator=g) + 1.0)}


def reference(inputs):
    import torch
    y = torch.exp(inputs["x"])
    m = y.amax(0, keepdim=True)
    return {"o": y - m, "m": m.reshape(-1)}


SPEC = KernelSpec(
    name="reduce_fold_bcast_and_store_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "m_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("x", IN, "float32", (R, C)), Arg("o", OUT, "float32", (R, C)), Arg("m", OUT, "float32", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    expect="pass",
)
