"""Two rank-5 loads with a wrapped axis: x folds into [2, 1024] carrying the wrap as digits, y (every
axis apart in DRAM) peels an uncut axis. Two pieces each, values checked. Breaks if p11 refuses a cut
of rank > 4 again, or folds or peels without carrying the cut digits (rule 18)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

D, N, W, O = 2, 128, 2, 1
X = D * D * D * D * N
Y = 2 * D * 2 * D * 2 * D * 2 * D * N


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_rank5_kernel(x_ptr, y_ptr, o_ptr, p_ptr, D: tl.constexpr, N: tl.constexpr,
                              W: tl.constexpr, O: tl.constexpr):
        i0 = tl.arange(0, D)[:, None, None, None, None]
        i1 = tl.arange(0, D)[None, :, None, None, None]
        i2 = tl.arange(0, D)[None, None, :, None, None]
        i3 = tl.arange(0, D)[None, None, None, :, None]
        n = tl.arange(0, N)[None, None, None, None, :]
        r1 = ((tl.arange(0, D) + O) % W)[None, :, None, None, None]
        r2 = ((tl.arange(0, D) + O) % W)[None, None, :, None, None]
        dst = i0 * (D * D * D * N) + i1 * (D * D * N) + i2 * (D * N) + i3 * N + n
        xs = i0 * (D * D * D * N) + r1 * (D * D * N) + i2 * (D * N) + i3 * N + n
        ys = (i0 * (2 * D * 2 * D * 2 * D * 2 * N) + i1 * (2 * D * 2 * D * 2 * N)
              + r2 * (2 * D * 2 * N) + i3 * (2 * N) + n)
        tl.store(o_ptr + dst, tl.load(x_ptr + xs) + 1.0)
        tl.store(p_ptr + dst, tl.load(y_ptr + ys) + 1.0)

    return ptr_wrap_rank5_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(X, generator=g, dtype=torch.float32),
            "y": torch.randn(Y, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = (torch.arange(D) + O) % W
    x = inputs["x"].reshape(D, D, D, D, N)
    y = inputs["y"].reshape(D, 2, D, 2, D, 2, D, 2 * N)[:, 0, :, 0, :, 0, :, :N]
    return {"o": (x[:, r] + 1.0).reshape(-1), "p": (y[:, :, r] + 1.0).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_wrap_rank5_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32", "p_ptr": "*fp32",
               "D": "constexpr", "N": "constexpr", "W": "constexpr", "O": "constexpr"},
    constexprs={"D": D, "N": N, "W": W, "O": O},
    args=[Arg("x", IN, "float32", (X,)), Arg("y", IN, "float32", (Y,)),
          Arg("o", OUT, "float32", (X,)), Arg("p", OUT, "float32", (X,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
