"""SiLU and GELU -- the two activations current LLM feed-forwards are made of.

    silu(x) = x * sigmoid(x)
    gelu(x) = 0.5x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

Llama and Mistral gate with SiLU, GPT and Gemma with GELU, and a feed-forward is
where most of a decoder layer's FLOPs are. `swiglu.py` puts SiLU in its real
setting -- gate, up, down -- and does not build; this file is the activation on
its own, so a failure there can be read as the activation or as the fusion
around it rather than as both at once.

WHAT IT PINS
  Neither one is a single instruction, and the whole chain has to stay in vector
  registers: a scalar libm call would lift each element into `fa0`, and there is
  one scalar register file for every lane (pipeline.py refuses the `vfmv.f.s` by
  name). That is exactly what stops ops_math_elementwise, whose
  `libdevice.atan2` has no vector form.

  tanh is written as `2*sigmoid(2u) - 1`, which is exact. This triton has no
  `tl.tanh` at all and `libdevice.tanh` would be the extern path, so the
  identity is not a shortcut -- it is the only spelling that stays vector.

  The exact GELU is deliberately NOT here. It needs `erf`, which is a libm call
  for the same reason, so it belongs with the blocked kernels and not with the
  two shapes that actually ship.

torch equivalent: F.silu(x), and F.gelu(x, approximate="tanh")

SiLU and tanh-GELU on their own, so a failure can be read apart from the fusion in swiglu.py. Inputs scaled by 3 to reach the saturating tails, where a wrong tanh stops looking like a rounding difference.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def activations_kernel(x_ptr, s_ptr, g_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.load(x_ptr + r)
        tl.store(s_ptr + r, x * tl.sigmoid(x))
        k = 0.7978845608028654                      # sqrt(2/pi)
        inner = k * (x + 0.044715 * x * x * x)
        # tanh(u) == 2*sigmoid(2u) - 1, exactly. This triton has no tl.tanh at
        # all, and libdevice.tanh would be an extern elementwise -- a scalar
        # libm call, which is the wall ops_math_elementwise stops at. The
        # identity keeps the whole chain in vector registers.
        th = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        tl.store(g_ptr + r, 0.5 * x * (1.0 + th))

    return activations_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32) * 3.0}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"s": torch.nn.functional.silu(x),
            "g": torch.nn.functional.gelu(x, approximate="tanh")}


SPEC = KernelSpec(
    name="activations_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*fp32", "g_ptr": "*fp32",
               "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("s", OUT, "float32", (N,)),
          Arg("g", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-5,
)
