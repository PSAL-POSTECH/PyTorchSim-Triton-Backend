"""An int8 matrix times an int8 vector the way Inductor writes it at one token: an int32 [X, R] tile of
products added to across a loop over K, summed over R after it. The loop carries one int32 a row instead
of the tile, each trip summing its products into it, as the float sum already does: integer addition is
exact in any order. Breaks if the int32 tile goes back to the scratchpad each trip."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, K, R = 256, 3072, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_int_loop_partials_into_rows_kernel(x_ptr, w_ptr, o_ptr, X: tl.constexpr, K: tl.constexpr,
                                                  R: tl.constexpr):
        rows = tl.arange(0, X)[:, None]
        r = tl.arange(0, R)[None, :]
        acc = tl.full([X, R], 0, tl.int32)
        for k in range(0, K, R):
            xv = tl.load(x_ptr + k + r).to(tl.int32)
            wv = tl.load(w_ptr + rows * K + k + r).to(tl.int32)
            acc = acc + xv * wv
        tl.store(o_ptr + rows, tl.sum(acc, 1)[:, None])

    return reduce_int_loop_partials_into_rows_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-128, 128, (K,), generator=g, dtype=torch.int8),
            "w": torch.randint(-128, 128, (X, K), generator=g, dtype=torch.int8)}


def reference(inputs):
    return {"o": (inputs["w"].int() @ inputs["x"].int()).reshape(X, 1)}


SPEC = KernelSpec(
    name="reduce_int_loop_partials_into_rows_kernel", kernel=kernel,
    signature={"x_ptr": "*i8", "w_ptr": "*i8", "o_ptr": "*i32", "X": "constexpr", "K": "constexpr",
               "R": "constexpr"},
    constexprs={"X": X, "K": K, "R": R},
    args=[Arg("x", IN, "int8", (K,)), Arg("w", IN, "int8", (X, K)), Arg("o", OUT, "int32", (X, 1))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
    expect_ir={"08-transformed.mlir": {"256x1024xi32": 0}},
)
