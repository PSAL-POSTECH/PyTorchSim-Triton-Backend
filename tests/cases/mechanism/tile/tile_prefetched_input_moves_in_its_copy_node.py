"""A kernel whose input w the graph copies into the scratchpad ahead of it (KernelSpec.prefetch): x and w are read a
block per trip, y written once. The TOG moves x and y only, and w's copy node loads w whole in one transfer. The
binary still moves all three, so the values are checked as ever. Breaks if a prefetched input's transfer reaches the
kernel's TOG, or its copy node is not written."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, T = 128, 256, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_prefetched_input_moves_in_its_copy_node(x_ptr, w_ptr, y_ptr, R: tl.constexpr, C: tl.constexpr,
                                                     T: tl.constexpr):
        offs = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        acc = tl.zeros((R, C), dtype=tl.float32)
        for t in range(T):
            acc += tl.load(x_ptr + t * R * C + offs) * tl.load(w_ptr + t * R * C + offs)
        tl.store(y_ptr + offs, acc)

    return tile_prefetched_input_moves_in_its_copy_node


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * R, C, generator=g), "w": torch.randn(T * R, C, generator=g)}


def reference(inputs):
    x, w = inputs["x"].reshape(T, R, C), inputs["w"].reshape(T, R, C)
    return {"y": (x * w).sum(0)}


SPEC = KernelSpec(
    name="tile_prefetched_input_moves_in_its_copy_node", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "y_ptr": "*fp32", "R": "constexpr", "C": "constexpr",
               "T": "constexpr"},
    constexprs={"R": R, "C": C, "T": T},
    args=[Arg("x", IN, "float32", (T * R, C)), Arg("w", IN, "float32", (T * R, C)), Arg("y", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    prefetch=(1,), expect_ir={"tog.cpp": {"togsim::load": 1, "togsim::store": 1},
                              "copy_node_1/tog.cpp": {"togsim::load": 1, "togsim::store": 0}},
    expect_stage="togcheck",
)
