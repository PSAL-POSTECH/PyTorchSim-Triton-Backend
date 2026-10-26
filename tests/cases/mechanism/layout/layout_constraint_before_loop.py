"""A 128x128 accumulator pinned before an scf.for (layout_constraint, lane_layout(1), on tl.zeros): the
pin's layout reaches the init, iter_arg, yield and result, so the loop body, its load and the store take
it, with no convert left. Breaks if propagation does not enter the loop from its init."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, M, N = 4, 128, 128


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import pytorchsim

    @triton.jit
    def layout_constraint_before_loop_kernel(x_ptr, o_ptr, K: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
        offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
        acc = pytorchsim.layout_constraint(tl.zeros((M, N), dtype=tl.float32), pytorchsim.lane_layout(1))
        for k in range(K):
            acc = acc + tl.load(x_ptr + k * M * N + offs)
        tl.store(o_ptr + offs, acc)

    return layout_constraint_before_loop_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(K, M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sum(0)}


SPEC = KernelSpec(
    name="layout_constraint_before_loop_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "K": "constexpr", "M": "constexpr", "N": "constexpr"},
    constexprs={"K": K, "M": M, "N": N},
    args=[Arg("x", IN, "float32", (K, M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"04-anchored.mlir": {"pytorchsim_extern.layout_constraint": 1},
               "05-layouts.mlir": {"pytorchsim_extern.layout_constraint": 0, "ttg.convert_layout": 0,
                                   "threadsPerWarp = [128, 1]": 0}},
)
