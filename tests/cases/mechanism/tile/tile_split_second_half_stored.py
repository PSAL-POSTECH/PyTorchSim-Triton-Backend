"""The second half of a tl.split stored straight out: a column that stays a view at bank offset 1,
so its transfer starts one element into its buffer. Breaks if the DMA address is taken from
the buffer's start without the view's offset: `a` is stored where `b` belongs."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
R = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_split_second_half_stored_kernel(x_ptr, o_ptr, R: tl.constexpr):
        r = tl.arange(0, R)
        x = tl.load(x_ptr + r[:, None] * 2 + tl.arange(0, 2)[None, :])
        a, b = tl.split(x)
        tl.store(o_ptr + r, b)

    return tile_split_second_half_stored_kernel


def make_inputs(torch, seed=0):
    return {"x": torch.arange(2 * R, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].view(R, 2)[:, 1].contiguous()}


SPEC = KernelSpec(
    name="tile_split_second_half_stored_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr"},
    constexprs={"R": R},
    args=[Arg("x", IN, "float32", (2 * R,)), Arg("o", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
