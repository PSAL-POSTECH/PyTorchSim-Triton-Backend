"""`(x % D) + D*(x // (H*D))` on ONE axis: RoPE's cos table broadcast over H heads, as Inductor
writes it once it flattens [T, H, D] to one index.

The two halves cut the axis at different places, D and H*D. D divides H*D, so their common
refinement is [N/(H*D), H, D] with strides [D, 0, 1]: the wrap D is whole laps of the outer
stride H*D, and the inner H*D then walks whole laps of D from a lap boundary -- a split, the heads
stride 0. Before, refineStateAxis took a wrap only equal to the new outer stride and the access
became a gather; the block start pid*BLOCK (a multiple of H*D) is what the grid of 2 exercises.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, H, D = 4, 4, 128
BLOCK = T * H * D // 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_rem_and_div_nest_on_one_axis_kernel(c_ptr, o_ptr, H: tl.constexpr, D: tl.constexpr,
                                                 BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(c_ptr + (x % D + D * (x // (H * D))))
        tl.store(o_ptr + x, v)

    return ptr_rem_and_div_nest_on_one_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"c": torch.randn(T * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    c = inputs["c"].reshape(T, 1, D)
    return {"o": c.expand(T, H, D).reshape(-1).clone()}


SPEC = KernelSpec(
    name="ptr_rem_and_div_nest_on_one_axis_kernel", kernel=kernel,
    signature={"c_ptr": "*fp32", "o_ptr": "*fp32", "H": "constexpr", "D": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"H": H, "D": D, "BLOCK": BLOCK},
    args=[Arg("c", IN, "float32", (T * D,)), Arg("o", OUT, "float32", (T * H * D,))],
    grid=(2,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.load": 1}},
)
