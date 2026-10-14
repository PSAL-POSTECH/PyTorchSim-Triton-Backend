"""MobileNetV2's inverted residual -- narrow, wide, narrow, and a linear exit.

    h = relu6(We @ x)        expand   c -> c*t
    y = Wp @ h  +  x         project  c*t -> c,  NO activation

BOTH NAMES IN "INVERTED RESIDUAL, LINEAR BOTTLENECK" CARRY AN ARGUMENT.

  INVERTED. A ResNet bottleneck goes wide -> narrow -> wide: reduce, do the
  expensive 3x3 in the narrow middle, expand back. This goes the other way, and
  the residual therefore connects the NARROW tensors. The claim is that the wide
  part is working space and the information lives in the narrow part -- which
  also makes the skip connection's memory `t` times smaller.

  LINEAR BOTTLENECK. The projecting 1x1 has no activation, and that is the
  paper's point rather than an omission. ReLU sets negatives to zero, which
  DESTROYS information; with many channels the lost part is probably still
  recorded in some other channel, but in a NARROW layer it is not and cannot be
  recovered. So nothing follows the layer that narrows.

WHAT IT ADDS HERE, AND THE GAP IT FOUND. The depthwise 3x3 between these two has
its own kernel (`depthwise_conv.py`); what is new in this one is the channel
structure -- TWO matmuls in a single kernel, with vector work between them.

  IT DOES NOT BUILD YET, and the reason is exactly that. Only one of the two
  reaches the matrix unit; the other survives lower_to_vcix and stage 4
  catches it ("a linalg.matmul survived lower_to_vcix"). Measured, so the
  diagnosis is not a guess:

    - both matmuls are present and correctly shaped at stage 3, each with its
      own `torchsim.systolic_tile` (the systolic lowering's, since removed);
    - the survivor is the SECOND one (the [32, 16] x [16, 32] project);
    - it is NOT about the buffer this kernel both writes and reads -- feeding
      the second matmul from a separate kernel argument instead makes no
      difference;
    - a one-trip K loop is not a workaround either: canonicalize folds it and
      the loopless tl.dot is declined for its own reason.

  So `lower_to_vcix` handles one matmul per kernel. That matters well beyond
  MobileNetV2 -- every transformer block chains two, and so does every one of
  MobileNetV2's own 17 blocks -- and it is why this file is here failing rather
  than trimmed down to something that passes.

torch equivalent: x + Wp @ relu6(We @ x)

GAP: two matmuls in one kernel, and only the first reaches the array. The second survives lower_to_vcix and stage 4 catches it; not caused by the shared buffer or by the loop shape.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, MID, HW = 32, 64, 32          # narrow, wide (t = 2 here), positions
BK = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def inverted_residual_kernel(x_ptr, we_ptr, wp_ptr, h_ptr, o_ptr,
                                 C: tl.constexpr, MID: tl.constexpr,
                                 HW: tl.constexpr, BK: tl.constexpr):
        c = tl.arange(0, C)
        mid = tl.arange(0, MID)
        hw = tl.arange(0, HW)
        k = tl.arange(0, BK)

        # expand: [MID, C] @ [C, HW]
        h = tl.zeros((MID, HW), dtype=tl.float32)
        for c0 in range(0, C, BK):
            we = tl.load(we_ptr + mid[:, None] * C + (c0 + k)[None, :])
            xs = tl.load(x_ptr + (c0 + k)[:, None] * HW + hw[None, :])
            h += tl.dot(we, xs)
        # THE WIDE TENSOR GOES OUT AND COMES BACK IN CHUNKS, which is not a
        # detour. A register tensor cannot be sliced by a loop-carried offset, so
        # the project's K loop has nothing to walk -- and a ONE-TRIP loop is not
        # a way out either: canonicalize folds it, leaving the loopless
        # tl.dot that lower_to_vcix declines. A real block has MID in the
        # hundreds and would not hold [MID, HW] in registers anyway.
        tl.store(h_ptr + mid[:, None] * HW + hw[None, :],
                 tl.minimum(tl.maximum(h, 0.0), 6.0))        # relu6

        # project: [C, MID] @ [MID, HW], and NO activation after it
        y = tl.zeros((C, HW), dtype=tl.float32)
        for m0 in range(0, MID, BK):
            wp = tl.load(wp_ptr + c[:, None] * MID + (m0 + k)[None, :])
            hs = tl.load(h_ptr + (m0 + k)[:, None] * HW + hw[None, :])
            y += tl.dot(wp, hs)
        tl.store(o_ptr + c[:, None] * HW + hw[None, :],
                 y + tl.load(x_ptr + c[:, None] * HW + hw[None, :]))

    return inverted_residual_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * HW, generator=g, dtype=torch.float32) * 0.3,
            "we": torch.randn(MID * C, generator=g, dtype=torch.float32) * 0.2,
            "wp": torch.randn(C * MID, generator=g, dtype=torch.float32) * 0.2}


def reference(inputs):
    x = inputs["x"].reshape(C, HW)
    we = inputs["we"].reshape(MID, C)
    wp = inputs["wp"].reshape(C, MID)
    h = (we @ x).clamp(0.0, 6.0)
    return {"h": h.reshape(-1), "o": (wp @ h + x).reshape(-1)}


SPEC = KernelSpec(
    name="inverted_residual_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "we_ptr": "*fp32", "wp_ptr": "*fp32",
               "h_ptr": "*fp32", "o_ptr": "*fp32", "C": "constexpr",
               "MID": "constexpr",
               "HW": "constexpr", "BK": "constexpr"},
    constexprs={"C": C, "MID": MID, "HW": HW, "BK": BK},
    args=[Arg("x", IN, "float32", (C * HW,)),
          Arg("we", IN, "float32", (MID * C,)),
          Arg("wp", IN, "float32", (C * MID,)),
          Arg("h", OUT, "float32", (MID * HW,)),
          Arg("o", OUT, "float32", (C * HW,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
