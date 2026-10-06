"""RoPE, interleaved -- GLM-4.5/4.6 and Cohere2 rotate ADJACENT dims, not halves.

    out[.., 2i]   = x[.., 2i] * cos[i] - x[.., 2i+1] * sin[i]
    out[.., 2i+1] = x[.., 2i+1] * cos[i] + x[.., 2i] * sin[i]

THE ROTATION IS THE SAME ROTATION. rope.py computes exactly these two lines. The
only difference is WHICH TWO ELEMENTS form pair i:

    rope.py (rotate_half)    j  and  j + D/2         two contiguous half-slices
    this file (interleaved)  2i and  2i + 1          two stride-2 slices

Both are a block-diagonal stack of D/2 planar rotations; they differ by a
permutation of the head dimension, and since the query and key projections are
learned, a model trained under one convention is simply a permuted version of a
model trained under the other. The math is identical. The MEMORY ACCESS IS NOT,
and that is the entire reason this file exists separately.

WHO USES WHICH, AND WHY IT IS NOT SETTLED
  The original RoFormer paper and every GPT-NeoX-derived checkpoint interleave.
  HuggingFace's Llama implementation instead permutes the weights at conversion
  time so it can use rotate_half, because two contiguous slices are cheaper on a
  GPU than two strided ones. Most of the field followed. GLM-4.5/4.6 and
  Cohere2's `rotate_half` is written with `x[..., 0::2]` / `x[..., 1::2]` and a
  `repeat_interleave(2)` on cos/sin, i.e. they kept the original convention, so
  an inference stack has to implement both. vLLM ships `is_neox_style` as a flag
  on its rotary kernel for exactly this.

WHAT IT ASKS OF THIS MACHINE THAT rope.py DOES NOT
  rope.py's docstring says it chose the halves form BECAUSE the interleaved one
  pairs neighbours a stride apart, and on a tile whose feature axis is the vector
  that is a read of every other element. This file is the other half of that
  sentence, asked deliberately.

    * The two loads are `2*h` and `2*h + 1` off the row base, so the inner
      offset advances by 2 per lane. That is still affine in `tl.arange`, so it
      should reach a strided `memref.subview` rather than a gather -- but a
      stride of 2 on the vector axis is a different lowering from a stride of 1
      at an offset, which is all rope.py ever produces. If the backend can only
      form unit-stride vector loads it has to fall back, and this is the smallest
      kernel that makes it.
    * The two STORES are strided the same way, interleaving the results back.
      rope.py's stores are two contiguous runs; here every write lands in a gap
      left by the other. A backend that fuses the pair into one full-width store
      has to reconstruct the interleave, which is a `vector.shuffle`, not a copy.
    * cos and sin are indexed by the PAIR index i, not by the element index, so
      they are [T, D/2] against a [T, D] output in both files -- but here the
      stretch is `repeat_interleave(2)` rather than `cat([c, c])`. Same operand,
      opposite expansion.

GLM applies rotary to only the first half of the head dim (partial rotary,
`rotary_percent = 0.5`) on top of this; that is an extra slice bound, orthogonal
to the pairing, and left out so the delta against rope.py stays exactly one
thing.

GLM-4.5/Cohere2: rope.py's rotation with stride-2 reads and interleaved stores instead of two contiguous half-slices. Same arithmetic, different walk through the head dim.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, D = 8, 32                       # positions, head dim (D/2 pairs)
H = D // 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rope_interleaved_pairs_kernel(x_ptr, cos_ptr, sin_ptr, o_ptr,
                                      T: tl.constexpr, D: tl.constexpr,
                                      H: tl.constexpr):
        t = tl.arange(0, T)
        h = tl.arange(0, H)

        # STRIDE 2 ALONG THE FEATURE AXIS. rope.py's two offsets differ by a
        # constant; these two differ by 1 while each advances by 2 per lane.
        ev = t[:, None] * D + (2 * h)[None, :]           # x[..., 0::2]
        od = ev + 1                                      # x[..., 1::2]

        x1 = tl.load(x_ptr + ev)
        x2 = tl.load(x_ptr + od)
        c = tl.load(cos_ptr + t[:, None] * H + h[None, :])
        s = tl.load(sin_ptr + t[:, None] * H + h[None, :])

        # The rotation itself is byte-for-byte rope.py's.
        tl.store(o_ptr + ev, x1 * c - x2 * s)
        tl.store(o_ptr + od, x2 * c + x1 * s)

    return rope_interleaved_pairs_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000.0 ** (torch.arange(H, dtype=torch.float32) / H))
    ang = pos * inv[None, :]
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32),
            "cos": ang.cos().reshape(-1), "sin": ang.sin().reshape(-1)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    c = inputs["cos"].reshape(T, H)
    s = inputs["sin"].reshape(T, H)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).reshape(T, D)
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="rope_interleaved_pairs_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "cos_ptr": "*fp32", "sin_ptr": "*fp32",
               "o_ptr": "*fp32", "T": "constexpr", "D": "constexpr",
               "H": "constexpr"},
    constexprs={"T": T, "D": D, "H": H},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("cos", IN, "float32", (T * H,)),
          Arg("sin", IN, "float32", (T * H,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
