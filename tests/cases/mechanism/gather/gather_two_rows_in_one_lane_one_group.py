"""Two gathered rows in one lane's bank, the bank short enough to be ONE register group: a [256, 32]
tile on 128 lanes is 64 elements a lane, one vrgather. Pins the single-instruction path of
gather_in_bank, where the chunk arithmetic of gather_by_groups never runs to carry a row base."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

L, D = 256, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_two_rows_in_one_lane_one_group_kernel(x_ptr, i_ptr, y_ptr, L: tl.constexpr,
                                                     D: tl.constexpr):
        at = tl.arange(0, L)[:, None] * D + tl.arange(0, D)[None, :]
        x = tl.exp(tl.load(x_ptr + at))
        idx = tl.load(i_ptr + at)
        tl.store(y_ptr + at, tl.gather(x, idx, 1))

    return gather_two_rows_in_one_lane_one_group_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(L, D, generator=g, dtype=torch.float32),
            "i": torch.randint(0, D, (L, D), generator=g, dtype=torch.int32)}


def reference(inputs):
    return {"y": inputs["x"].exp().gather(1, inputs["i"].long())}


SPEC = KernelSpec(
    name="gather_two_rows_in_one_lane_one_group_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "y_ptr": "*fp32", "L": "constexpr", "D": "constexpr"},
    constexprs={"L": L, "D": D},
    args=[Arg("x", IN, "float32", (L, D)), Arg("i", IN, "int32", (L, D)),
          Arg("y", OUT, "float32", (L, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
