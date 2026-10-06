"""RoPE -- rotary position embedding, the only positional scheme in current LLMs.

    out[.., 2i]   = x[.., 2i] * cos - x[.., 2i+1] * sin
    out[.., 2i+1] = x[.., 2i+1] * cos + x[.., 2i] * sin

Llama, Mistral, Qwen and Gemma all use this and no learned position table, so it
runs on every token of every layer. It is also the one LLM op whose data
movement is a PERMUTATION rather than a reduction or a broadcast, which is what
makes it interesting here.

WHY IT IS THE HARD SHAPE FOR THIS MACHINE
  The rotation pairs element 2i with 2i+1, so it needs two values that are
  adjacent along the feature axis at the same time. This file uses the HALVES
  split that HuggingFace's implementation uses -- pair j with j + D/2 -- because
  the interleaved form would pair neighbours a stride apart, and on a tile whose
  feature axis is the vector that is a strided read of every other element.

  The halves form is two contiguous slices, which is `tensor.extract_slice` at
  offsets 0 and D/2 along the INNER axis. That is unit-stride, so it stays a
  vector read; the pieces then go back side by side, which is the concat shape.

  `cos` and `sin` are per (position, feature) and shared across heads, so they
  broadcast along the row -- the free direction once dim 0 is on the lanes.

torch equivalent: the rotate_half form, x1*cos - x2*sin || x2*cos + x1*sin

The rotate_half form, because the interleaved one pairs neighbours a stride apart and that is a strided read of the vector axis.
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
    def rope_kernel(x_ptr, cos_ptr, sin_ptr, o_ptr,
                    T: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
        t = tl.arange(0, T)
        h = tl.arange(0, H)
        lo = t[:, None] * D + h[None, :]                 # x[.., :D/2]
        hi = lo + H                                      # x[.., D/2:]

        x1 = tl.load(x_ptr + lo)
        x2 = tl.load(x_ptr + hi)
        c = tl.load(cos_ptr + t[:, None] * H + h[None, :])
        s = tl.load(sin_ptr + t[:, None] * H + h[None, :])

        tl.store(o_ptr + lo, x1 * c - x2 * s)
        tl.store(o_ptr + hi, x2 * c + x1 * s)

    return rope_kernel


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
    x1, x2 = x[:, :H], x[:, H:]
    return {"o": torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=1).reshape(-1)}


SPEC = KernelSpec(
    name="rope_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "cos_ptr": "*fp32", "sin_ptr": "*fp32",
               "o_ptr": "*fp32", "T": "constexpr", "D": "constexpr",
               "H": "constexpr"},
    constexprs={"T": T, "D": D, "H": H},
    args=[Arg("x", IN, "float32", (T * D,)), Arg("cos", IN, "float32", (T * H,)),
          Arg("sin", IN, "float32", (T * H,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
