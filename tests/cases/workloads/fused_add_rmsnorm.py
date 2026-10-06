"""vLLM's `fused_add_rms_norm` -- the residual add and the norm, returning BOTH.

torch equivalent:
    h = x + residual                       # the new residual, kept
    y = h * rsqrt(mean(h*h) + eps) * w      # the normed activation

WHY THIS EXACT PAIR IS ONE KERNEL EVERYWHERE
  A transformer block is `h = h + attn(norm(h))` then `h = h + mlp(norm(h))`, so
  every single sublayer boundary is an add followed by a norm, and the ADD'S
  RESULT IS NEEDED AGAIN as the next residual. Two kernels would write h, read
  it back, and write y -- three passes over the hidden state for arithmetic that
  costs one. vLLM, TensorRT-LLM and Liger all ship this fused, and it runs twice
  per layer, so on a 32-layer model it is 64 launches saved and, more to the
  point, 64 round-trips of the largest tensor in the model.

WHAT MAKES IT MORE THAN AN ELEMENTWISE CHAIN
  TWO OUTPUTS FROM ONE TILE, and they are at different points in the chain: `h`
  comes out before the reduction, `y` after it. So the tile has to stay live
  across a reduction and a broadcast rather than being consumed and discarded.
  A backend that streams elementwise work through a reduction has to hold the
  pre-reduction value somewhere, which is the pressure this kernel applies.

  RMSNorm and not LayerNorm because that is what current models use, and it
  matters here: there is no mean to subtract, so the ONLY reduction is the sum
  of squares and the dependency is exactly one deep.

The weight is per-feature and broadcasts along the row axis, the ordinary
direction. eps is inside the sqrt, not added after, which is where the numbers
differ if it is moved.

Two outputs straddling a reduction. `h` is exact; `y` carries one rsqrt, so a mismatch there is arithmetic and a mismatch in `h` is a lowering fault.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, H = 128, 64                          # tokens, hidden
EPS = 1e-6


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fused_add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, h_ptr, y_ptr,
                                 T: tl.constexpr, H: tl.constexpr,
                                 EPS: tl.constexpr):
        rt = tl.arange(0, T)
        rh = tl.arange(0, H)
        off = rt[:, None] * H + rh[None, :]

        # The add's result is an OUTPUT and the norm's input at once.
        h = tl.load(x_ptr + off) + tl.load(res_ptr + off)
        tl.store(h_ptr + off, h)

        inv = 1.0 / tl.sqrt(tl.sum(h * h, axis=1) / H + EPS)
        tl.store(y_ptr + off, h * inv[:, None] * tl.load(w_ptr + rh)[None, :])

    return fused_add_rmsnorm_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T, H, generator=g, dtype=torch.float32),
            "res": torch.randn(T, H, generator=g, dtype=torch.float32),
            "w": torch.randn(H, generator=g, dtype=torch.float32)}


def reference(inputs):
    h = inputs["x"] + inputs["res"]
    inv = (h.pow(2).mean(dim=1, keepdim=True) + EPS).rsqrt()
    return {"h": h, "y": h * inv * inputs["w"]}


SPEC = KernelSpec(
    name="fused_add_rmsnorm_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "res_ptr": "*fp32", "w_ptr": "*fp32",
               "h_ptr": "*fp32", "y_ptr": "*fp32",
               "T": "constexpr", "H": "constexpr", "EPS": "constexpr"},
    constexprs={"T": T, "H": H, "EPS": EPS},
    args=[Arg("x", IN, "float32", (T, H)),
          Arg("res", IN, "float32", (T, H)),
          Arg("w", IN, "float32", (H,)),
          Arg("h", OUT, "float32", (T, H)),
          Arg("y", OUT, "float32", (T, H))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
