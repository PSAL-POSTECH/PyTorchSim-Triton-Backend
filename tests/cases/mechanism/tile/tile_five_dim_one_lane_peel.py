"""A ONE-LANE tile filling all four descriptor dims has no extent-1 dim left to
name as its lane axis, so slot 0 named a real one and Spike returned 69.6226 for
a reference of 0.2109. Breaks it: assuming slot 0 is free instead of choosing.

IT WAS REFUSED UNTIL THE CROSSBAR'S REDUCE RETIRED. The fold that put this tile in one bank is now a transpose and the lane's own reduction, so the tile keeps an axis and the descriptor keeps an extent-1 dim to name. The paragraph below is why the refusal existed. THE DESCRIPTOR HAS NO ENCODING FOR 'NO SPLIT'. It is 4-D whatever the tile's rank and the DMA spreads shape4[vsa] over the lanes, so one lane is spelled by naming a dim of EXTENT 1. Padding a tile of rank < 4 up to four mints one, which is why the slot was the constant 0 and why it was right 113 times out of 113 across 158 stage-5 modules: every one-lane transfer in the suite is rank <= 3. THIS TILE IS THE 114th. Reducing all five axes leaves memref<2x2x2x2x2xf32, 1> stamped ONE_LANE (-2), the peel hands the descriptor four dims of extent 2, and slot 0 then named a REAL axis that the DMA duly spread over two lanes. Nothing could see it -- the descriptor verifies, the asm verifies, Spike runs -- and the answer came back 69.6226 against a reference of 0.2109 (rule 12). WHY REFUSE RATHER THAN PEEL ONE MORE AXIS to mint the slot: that was built and measured. It lowers, and the answer is still wrong -- 33.5498, a second defect further in on a path no kernel here needs. A refusal that names the shape beats a number nobody can check. WHAT BREAKS IT: assuming slot 0 is free instead of choosing an extent-1 dim, or lowering this shape at all before the 33.5498 is understood.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", ".."))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT  # noqa: E402

D = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def five_dim_one_lane_peel_kernel(x_ptr, o_ptr, D: tl.constexpr):
        i0 = tl.arange(0, D)
        i1 = tl.arange(0, D)
        i2 = tl.arange(0, D)
        i3 = tl.arange(0, D)
        i4 = tl.arange(0, D)
        off = (i0[:, None, None, None, None] * (D * D * D * D)
               + i1[None, :, None, None, None] * (D * D * D)
               + i2[None, None, :, None, None] * (D * D)
               + i3[None, None, None, :, None] * D
               + i4[None, None, None, None, :])
        x = tl.load(x_ptr + off)
        s = tl.sum(tl.sum(tl.sum(tl.sum(tl.sum(x, 4), 3), 2), 1), 0)
        tl.store(o_ptr + tl.arange(0, 1), s[None])

    return five_dim_one_lane_peel_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(D ** 5, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sum().reshape(1)}


SPEC = KernelSpec(
    name="five_dim_one_lane_peel_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "D": "constexpr"},
    constexprs={"D": D},
    args=[Arg("x", IN, "float32", (D ** 5,)),
          Arg("o", OUT, "float32", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
