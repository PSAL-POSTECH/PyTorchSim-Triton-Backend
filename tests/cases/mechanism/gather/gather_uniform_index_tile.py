"""A [1, R] gather through a [1, R] index tile, both UNIFORM (one bank, copied to every lane), weighing
128 rows: pins the timing stage. Breaks it: a TOG node stating the logical R elements of the gather
while its descriptor walks lanes x R, so TOGSim drops Spike's index dump (old tree: "found no index dump")."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, R, V = 128, 8, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_uniform_index_tile_kernel(i_ptr, t_ptr, w_ptr, o_ptr,
                                         X: tl.constexpr, R: tl.constexpr):
        r = tl.arange(0, R)[None, :]
        x = tl.arange(0, X)[:, None]
        g = tl.load(t_ptr + tl.load(i_ptr + r))
        w = tl.load(w_ptr + x * R + r)
        tl.store(o_ptr + tl.arange(0, X), tl.sum(w * g, 1))

    return gather_uniform_index_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"i": torch.randperm(V, generator=g)[:R].to(torch.int64),
            "t": torch.randn(V, generator=g, dtype=torch.float32),
            "w": torch.randn(X * R, generator=g, dtype=torch.float32)}


def reference(inputs):
    g = inputs["t"][inputs["i"]]
    return {"o": (inputs["w"].view(X, R) * g[None, :]).sum(1)}


SPEC = KernelSpec(
    name="gather_uniform_index_tile_kernel", kernel=kernel,
    signature={"i_ptr": "*i64", "t_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "X": "constexpr", "R": "constexpr"},
    constexprs={"X": X, "R": R},
    args=[Arg("i", IN, "int64", (R,)), Arg("t", IN, "float32", (V,)),
          Arg("w", IN, "float32", (X * R,)), Arg("o", OUT, "float32", (X,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
    expect_stage="timing",
)
