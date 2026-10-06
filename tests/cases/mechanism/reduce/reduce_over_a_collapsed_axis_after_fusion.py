"""A row mean and variance over an axis triton-shared split and collapsed back
([X, 2, 256] -> [X, 512]): fusion after layout must not fold the collapse through
the reduce, which re-opens it as a fold over two axes vectorize cannot take.

Stable Diffusion's group norm in its smallest form: a per-group bias read by `col // 256` splits the row, the sum folds it collapsed. Fusion folding the collapse through the reduce made it a fold over [1, 2], which vectorize refuses.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, R, G = 32, 512, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_over_a_collapsed_axis_after_fusion_kernel(
            x_ptr, b_ptr, mean_ptr, m2_ptr,
            X: tl.constexpr, R: tl.constexpr, G: tl.constexpr):
        rows = tl.arange(0, X)[:, None]
        cols = tl.arange(0, R)[None, :]
        grp = cols // (R // G)
        x = tl.load(x_ptr + (cols + R * rows))
        b = tl.load(b_ptr + (grp + G * rows))
        h = x + b
        mean = tl.sum(h, 1)[:, None] / R
        d = h - mean
        tl.store(mean_ptr + rows, mean)
        tl.store(m2_ptr + rows, tl.sum(d * d, 1)[:, None])

    return reduce_over_a_collapsed_axis_after_fusion_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(X * R, generator=g, dtype=torch.float32),
            "b": torch.randn(X * G, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(X, G, R // G)
    h = (x + inputs["b"].reshape(X, G, 1)).reshape(X, R)
    mean = h.mean(dim=1)
    return {"mean": mean, "m2": ((h - mean[:, None]) ** 2).sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_over_a_collapsed_axis_after_fusion_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*fp32", "mean_ptr": "*fp32",
               "m2_ptr": "*fp32", "X": "constexpr", "R": "constexpr",
               "G": "constexpr"},
    constexprs={"X": X, "R": R, "G": G},
    args=[Arg("x", IN, "float32", (X * R,)), Arg("b", IN, "float32", (X * G,)),
          Arg("mean", OUT, "float32", (X,)), Arg("m2", OUT, "float32", (X,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
