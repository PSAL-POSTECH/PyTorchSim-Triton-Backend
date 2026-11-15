"""A kernel whose input x and output y the graph keeps in the scratchpad (KernelSpec.on_chip): x is read a block
per trip beside w, y written once. The TOG moves only w: no load of x, no store of y, and the wait the loads of x
and w share stays for w. The binary still moves all three, so the values are checked as ever. Breaks if an
on-chip argument's transfer reaches the TOG, or its wait is dropped with it."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, T = 128, 256, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_on_chip_arguments_move_no_dma(x_ptr, w_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr,
                                           T: tl.constexpr):
        offs = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        acc = tl.zeros((R, C), dtype=tl.float32)
        for t in range(T):
            acc += tl.load(x_ptr + t * R * C + offs) * tl.load(w_ptr + t * R * C + offs)
        tl.store(y_ptr + offs, acc)

    return tile_on_chip_arguments_move_no_dma


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * R, C, generator=g), "w": torch.randn(T * R, C, generator=g)}


def reference(inputs):
    x, w = inputs["x"].reshape(T, R, C), inputs["w"].reshape(T, R, C)
    return {"y": (x * w).sum(0)}


SPEC = KernelSpec(
    name="tile_on_chip_arguments_move_no_dma", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "y_ptr": "*fp32", "R": "constexpr", "C": "constexpr",
               "T": "constexpr"},
    constexprs={"R": R, "C": C, "T": T},
    args=[Arg("x", IN, "float32", (T * R, C)), Arg("w", IN, "float32", (T * R, C)), Arg("y", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    on_chip=(0, 2), expect_ir={"tog.cpp": {"togsim::load": 1, "togsim::wait": 1, "togsim::store": 0}},
    expect_stage="togcheck",
)
