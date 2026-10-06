"""The pairwise NxM IoU matrix -- the inner loop of NMS, and of every matcher.

    lt    = max(a[:, None, :2], b[None, :, :2])
    rb    = min(a[:, None, 2:], b[None, :, 2:])
    inter = clamp(rb - lt, min=0).prod
    iou   = inter / (area_a[:, None] + area_b[None, :] - inter)

torch equivalent: torchvision.ops.box_iou(a, b)

WHERE IT RUNS. Non-max suppression is O(N^2) IoU evaluations and it is the last
thing between a detector's raw output and its predictions -- on a YOLO head that
is thousands of surviving boxes against each other, per image, per class. The
same matrix is the cost term in DETR's and RT-DETR's Hungarian matcher (as GIoU)
and the assignment criterion in every anchor-based trainer. It is also the part
of a detector that people keep having to write a custom kernel for, because
`torchvision.ops.nms` is a CUDA extension and nothing else.

  yolo_dfl.py produces the boxes; this consumes them. Between the two there is
  nothing left of the network.

WHAT IT ASKS THAT NOTHING ELSE HERE DOES: A PAIRWISE TILE WITH NO CONTRACTION.

  Every other NxM tile in this suite is either a matmul -- where the pair axis
  is produced by `tl.dot` and immediately reduced away -- or built from INDEX
  ARITHMETIC, as in alibi.py (`i - j`) and t5_rel_bias.py (a bucketed `j - i`),
  where both operands are the lane index and nothing is loaded. Here the tile is
  the OUTER COMBINATION OF TWO DIFFERENT LOADED VECTORS, with `max` and `min`
  where a matmul would have `*` and `+`, and the result is the ANSWER: it is
  never folded. So the (N, M) shape has to be materialised, and the backend
  cannot borrow the systolic array's layout for it.

  N != M on purpose. A square tile lets a wrong broadcast direction still
  typecheck and still produce plausible numbers -- `a[:, None]` and `a[None, :]`
  are indistinguishable at 32x32. At 16x32 they are not.

THE LOADS ARE AN ARRAY-OF-STRUCTS READ. Boxes are stored [N, 4] xyxy, so each of
the eight coordinate vectors is a STRIDE-4 rank-1 gather out of one buffer --
eight of them from two pointers. That is the deinterleave `qkv_split.py` does
with a matrix; here it is the degenerate rank-1 case, and it is how detection
data actually arrives.

CLAMP AT ZERO IS NOT COSMETIC. Disjoint boxes give `rb - lt` negative on one or
both axes, and without the clamp the product of two negatives is a POSITIVE
intersection -- boxes that do not touch would report overlap, and NMS would
delete them. The clamp is on each side separately, before the product, which is
the only order that works.

No epsilon on the union: both areas are positive for a well-formed box, so the
denominator is bounded below by the larger area.

NMS's inner loop: an NxM pairwise tile built from two loaded rank-1 families by min/max instead of a matmul, never reduced, and non-square so a flipped broadcast cannot hide.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 16                          # boxes on the left
M = 32                          # boxes on the right -- deliberately different


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def box_iou_kernel(a_ptr, b_ptr, o_ptr, N: tl.constexpr, M: tl.constexpr):
        n = tl.arange(0, N)
        m = tl.arange(0, M)

        # [N, 4] and [M, 4] xyxy: every coordinate vector is a stride-4 read.
        ax1 = tl.load(a_ptr + n * 4 + 0)
        ay1 = tl.load(a_ptr + n * 4 + 1)
        ax2 = tl.load(a_ptr + n * 4 + 2)
        ay2 = tl.load(a_ptr + n * 4 + 3)
        bx1 = tl.load(b_ptr + m * 4 + 0)
        by1 = tl.load(b_ptr + m * 4 + 1)
        bx2 = tl.load(b_ptr + m * 4 + 2)
        by2 = tl.load(b_ptr + m * 4 + 3)

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        # The pairwise tile, and the only place the two families meet.
        lx = tl.maximum(ax1[:, None], bx1[None, :])
        ly = tl.maximum(ay1[:, None], by1[None, :])
        rx = tl.minimum(ax2[:, None], bx2[None, :])
        ry = tl.minimum(ay2[:, None], by2[None, :])

        # Clamp each side BEFORE the product: two negatives would multiply to a
        # positive intersection and disjoint boxes would look overlapped.
        iw = tl.maximum(rx - lx, 0.0)
        ih = tl.maximum(ry - ly, 0.0)
        inter = iw * ih

        tl.store(o_ptr + n[:, None] * M + m[None, :],
                 inter / (area_a[:, None] + area_b[None, :] - inter))

    return box_iou_kernel


def _boxes(torch, g, k):
    """k well-formed xyxy boxes, sized so roughly half the pairs overlap --
    enough non-zero IoUs to test the arithmetic and enough disjoint pairs to
    test the clamp."""
    xy = torch.rand(k, 2, generator=g, dtype=torch.float32) * 12.0
    wh = torch.rand(k, 2, generator=g, dtype=torch.float32) * 8.0 + 2.0
    return torch.cat([xy, xy + wh], dim=1)


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": _boxes(torch, g, N).reshape(-1).contiguous(),
            "b": _boxes(torch, g, M).reshape(-1).contiguous()}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(N, 4)
    b = inputs["b"].reshape(M, 4)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    iou = inter / (area_a[:, None] + area_b[None, :] - inter)
    return {"o": iou.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="box_iou_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr", "M": "constexpr"},
    constexprs={"N": N, "M": M},
    args=[Arg("a", IN, "float32", (N * 4,)),
          Arg("b", IN, "float32", (M * 4,)),
          Arg("o", OUT, "float32", (N * M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
