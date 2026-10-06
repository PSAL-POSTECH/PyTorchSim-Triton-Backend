"""`F.grid_sample` -- bilinear sampling where the COORDINATES ARE DATA.

torch equivalent: F.grid_sample(x, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)

THE DIFFERENCE FROM upsample.py, WHICH IS THE WHOLE POINT. That kernel is also
four gathers and three lerps, but its source coordinate is `oy * 0.5 - 0.25` --
an affine function of the lane index, known at compile time up to a constant.
PtrAnalysis walks straight through it and the read stays a strided descriptor.
Here the coordinate is LOADED FROM MEMORY. `y0` is a tensor whose value nothing
upstream knows, so `y0 * W + x0` is an address that has to be materialised per
lane and the load is an honest gather -- one that cannot be turned back into a
block descriptor by any amount of analysis.

  That is the line this file draws: a fixed-ratio resize and an arbitrary warp
  are the same arithmetic and completely different memory. A backend that
  handles `upsample` by pattern-matching the resize will not handle this.

WHY IT IS THE MOST VALUABLE MISSING PRIMITIVE. It is not one op, it is the
bottom of three:

    DCNv2                 conv taps displaced by learned per-pixel offsets
    deformable attention  a few learned points per query, per feature level
    SAM2 / optical flow   warping a feature map by a predicted field

all of which are `grid_sample` with a different producer for the grid. Nothing
in the suite covers it, so all three are uncovered at once.

THE ADDRESS ARITHMETIC IS THE TEST.
  align_corners=False:  ix = ((gx + 1) * W - 1) / 2
  This is the half-pixel convention -- the same one upsample.py uses and for the
  same reason, that it is the only one under which sampling is composable.

  `tl.floor` AND NOT A CAST. ix goes negative whenever the grid points off the
  left edge, and an int cast truncates toward zero, so `floor(-0.3) = -1` but
  `int(-0.3) = 0`. That single-pixel error only shows up on the border lanes,
  which is exactly where a warp spends its time.

  ZERO PADDING IS PER CORNER, not a guard on the sample. torch adds a corner's
  contribution only `if within_bounds(iy, ix)`, so a point half off the edge
  gets the in-bounds corners' weights and nothing else -- the result FADES at
  the boundary rather than clamping. Implemented here as `tl.load(..., mask=,
  other=0.0)` with the index clamped as well, because the mask decides the
  value but the address is still formed.

The plane is single-channel and the tile is [OH, OW]: the output grid IS the
tile. ms_deform_attn.py does the same gather with the queries flattened and a
channel axis instead, which is the other shape this primitive appears in.

The warp under DCNv2, deformable attention and SAM2. Same lerps as upsample.py, but the source index is loaded rather than affine in the lane, so the read is a true gather and the padding is per corner.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

H, W = 16, 16                   # source plane
OH, OW = 8, 8                   # output grid == the tile


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def grid_sample_kernel(x_ptr, g_ptr, o_ptr,
                           H: tl.constexpr, W: tl.constexpr,
                           OH: tl.constexpr, OW: tl.constexpr):
        oy = tl.arange(0, OH)
        ox = tl.arange(0, OW)
        q = oy[:, None] * OW + ox[None, :]

        # The grid is [OH, OW, 2] with x before y, so the two fields are read
        # with stride 2 from the same buffer. This is the load that makes the
        # rest of the kernel data-dependent.
        gx = tl.load(g_ptr + q * 2 + 0)
        gy = tl.load(g_ptr + q * 2 + 1)

        # align_corners=False: half-pixel aligned unnormalisation.
        ix = ((gx + 1.0) * W - 1.0) * 0.5
        iy = ((gy + 1.0) * H - 1.0) * 0.5

        # floor, not a cast -- ix is negative wherever the grid points off the
        # left edge and a cast truncates toward zero instead of down.
        fx0 = tl.floor(ix)
        fy0 = tl.floor(iy)
        wx = ix - fx0
        wy = iy - fy0
        x0 = fx0.to(tl.int32)
        y0 = fy0.to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # Zero padding is decided per CORNER, so each has its own mask; the
        # index is clamped too because the address is formed either way.
        okx0 = (x0 >= 0) & (x0 < W)
        okx1 = (x1 >= 0) & (x1 < W)
        oky0 = (y0 >= 0) & (y0 < H)
        oky1 = (y1 >= 0) & (y1 < H)
        cx0 = tl.minimum(tl.maximum(x0, 0), W - 1)
        cx1 = tl.minimum(tl.maximum(x1, 0), W - 1)
        cy0 = tl.minimum(tl.maximum(y0, 0), H - 1)
        cy1 = tl.minimum(tl.maximum(y1, 0), H - 1)

        v00 = tl.load(x_ptr + cy0 * W + cx0, mask=oky0 & okx0, other=0.0)
        v01 = tl.load(x_ptr + cy0 * W + cx1, mask=oky0 & okx1, other=0.0)
        v10 = tl.load(x_ptr + cy1 * W + cx0, mask=oky1 & okx0, other=0.0)
        v11 = tl.load(x_ptr + cy1 * W + cx1, mask=oky1 & okx1, other=0.0)

        # The weights come from the UNCLAMPED coordinate, which is what makes a
        # half-off-the-edge sample fade instead of clamp.
        out = (v00 * ((1.0 - wy) * (1.0 - wx)) + v01 * ((1.0 - wy) * wx)
               + v10 * (wy * (1.0 - wx)) + v11 * (wy * wx))
        tl.store(o_ptr + q, out)

    return grid_sample_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # The grid deliberately overshoots [-1, 1] so the zero-padding corners fire.
    grid = (torch.rand(OH * OW * 2, generator=g, dtype=torch.float32)
            * 2.4 - 1.2)
    return {"x": torch.randn(H * W, generator=g, dtype=torch.float32),
            "g": grid}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(1, 1, H, W)
    grid = inputs["g"].reshape(1, OH, OW, 2)
    o = torch.nn.functional.grid_sample(x, grid, mode="bilinear",
                                        padding_mode="zeros",
                                        align_corners=False)
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="grid_sample_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "g_ptr": "*fp32", "o_ptr": "*fp32",
               "H": "constexpr", "W": "constexpr",
               "OH": "constexpr", "OW": "constexpr"},
    constexprs={"H": H, "W": W, "OH": OH, "OW": OW},
    args=[Arg("x", IN, "float32", (H * W,)),
          Arg("g", IN, "float32", (OH * OW * 2,)),
          Arg("o", OUT, "float32", (OH * OW,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
