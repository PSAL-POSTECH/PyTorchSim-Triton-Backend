"""GPT-OSS `_apply_gate` -- a biased, clamped, interleaved GLU with a residual +1.

    gate = h[..., 0::2] + b[0::2]          gate and up are INTERLEAVED, not halves
    up   = h[..., 1::2] + b[1::2]
    gate = min(gate, LIMIT)                ONE-SIDED clamp
    up   = clamp(up, -LIMIT, LIMIT)        two-sided
    out  = (up + 1) * gate * sigmoid(ALPHA * gate)

ALPHA = 1.702, LIMIT = 7.0, and the bias is per expert.

EVERY ONE OF THOSE FOUR DEVIATIONS IS DELIBERATE, and swiglu.py has none of them.

  THE +1 ON `up`. A SwiGLU unit computes `up * silu(gate)`, so when the up
  projection is near zero the unit outputs nothing regardless of the gate.
  `(up + 1) * glu` makes the up path a MULTIPLICATIVE CORRECTION around unity:
  the default behaviour is to pass the gated value through, and `up` modulates
  it. It is a residual placed inside the nonlinearity rather than around it, and
  it changes what the layer does at initialisation -- an untrained up projection
  gives `glu`, not zero.

  ALPHA = 1.702. `x * sigmoid(1.702x)` is the standard sigmoid-linear
  approximation to GELU; at alpha = 1 it is SiLU. So GPT-OSS's gate
  nonlinearity is a GELU-shaped curve expressed in SiLU's algebra, and the
  constant lives inside the exponent -- which means it cannot be hoisted out or
  folded into the preceding matmul, because sigmoid is not homogeneous.

  THE CLAMPS ARE ASYMMETRIC, and this is the detail an implementation gets
  wrong. `gate` is clamped ABOVE ONLY: large positive gate saturates the
  sigmoid, so capping it costs nothing and prevents the fp16 overflow that a
  runaway expert produces; large NEGATIVE gate is already harmless because
  `x*sigmoid(x)` decays to zero there, and clamping it below would flatten the
  curve where it is still doing work. `up` is clamped on BOTH sides because it
  multiplies linearly and has no saturation of its own. Two clamps, three
  bounds, and getting the fourth bound "for symmetry" changes the function.

  GATE AND UP ARE INTERLEAVED IN MEMORY. swiglu.py runs two separate matmuls
  into two separate tensors. GPT-OSS's expert projection is one fused weight
  whose output columns alternate gate, up, gate, up -- so the split is a stride-2
  read of one tile, the same access shape rope_interleaved_pairs.py has, arriving
  here for a completely different reason. The bias is interleaved with it.

WHAT IT ASKS OF THE BACKEND
  Two stride-2 loads off one row plus a stride-2 bias load, a min, a clamp, an
  exp on the result, and an elementwise join -- all inside a static loop over
  experts, so the per-expert bias stays a rank-1 operand rather than becoming an
  axis of the tile. swiglu.py's pressure was two independently pinned matmul
  outputs having to agree; this file has no matmul at all. It is what happens
  AFTER one, and the question is whether a strided pair of slices off a single
  tile survives being fed through a transcendental and rejoined.

  The sigmoid is written out as `1 / (1 + exp(-a*g))` for swiglu.py's reason: an
  arith + math.exp chain reaches the backend, not an extern call whose lowering
  is a separate question. The clamps are `tl.minimum` / `tl.maximum`, which is
  what `torch.clamp` lowers to anyway.

LAYOUT. `h` is [E, T, 2F]: tokens already grouped by expert, which is what a
fused MoE kernel sees after the routing sort (moe_router.py / fused_moe_expert_mlp
cover that half). The expert loop is `tl.static_range` so `b[e]` is a rank-1
tile. Inputs are scaled so both clamps genuinely fire -- a test where LIMIT is
never reached tests nothing.

GPT-OSS: interleaved gate/up off one fused projection, a per-expert bias, asymmetric clamps, alpha inside the exponent, and a +1 that makes `up` a correction rather than a gate. swiglu.py has none of it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

E, T, F = 2, 16, 16        # experts, tokens per expert, ffn width
F2 = 2 * F                 # the fused gate/up projection, interleaved
ALPHA = 1.702              # GELU-shaped, not SiLU's 1.0
LIMIT = 7.0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gptoss_clamped_glu_kernel(h_ptr, b_ptr, o_ptr, E: tl.constexpr,
                                  T: tl.constexpr, F: tl.constexpr,
                                  F2: tl.constexpr, ALPHA: tl.constexpr,
                                  LIMIT: tl.constexpr):
        rt = tl.arange(0, T)
        rf = tl.arange(0, F)
        ev = (2 * rf)[None, :]          # gate columns
        od = ev + 1                     # up columns

        for e in tl.static_range(E):
            row = e * T * F2 + rt[:, None] * F2
            bb = e * F2

            # Stride-2 split of ONE fused projection, bias interleaved with it.
            gate = tl.load(h_ptr + row + ev) + tl.load(b_ptr + bb + ev)
            up = tl.load(h_ptr + row + od) + tl.load(b_ptr + bb + od)

            # ASYMMETRIC. gate above only; up on both sides.
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)

            # ALPHA is inside the exponent, so it cannot be folded into the
            # matmul that produced `gate` -- sigmoid is not homogeneous.
            glu = gate / (1.0 + tl.exp(-ALPHA * gate))

            # The +1: `up` corrects around unity instead of gating from zero.
            tl.store(o_ptr + e * T * F + rt[:, None] * F + rf[None, :],
                     (up + 1.0) * glu)

    return gptoss_clamped_glu_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # std 4 with a bias of the same order, so a good fraction of both tensors
    # lands outside [-7, 7] and each clamp bound is actually exercised.
    return {"h": torch.randn(E, T, F2, generator=g, dtype=torch.float32) * 4.0,
            "b": torch.randn(E, F2, generator=g, dtype=torch.float32) * 2.0}


def reference(inputs):
    import torch
    h = inputs["h"]
    b = inputs["b"]
    gate = h[:, :, 0::2] + b[:, None, 0::2]
    up = h[:, :, 1::2] + b[:, None, 1::2]
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(ALPHA * gate)
    return {"o": ((up + 1.0) * glu).contiguous()}


SPEC = KernelSpec(
    name="gptoss_clamped_glu_kernel", kernel=kernel,
    signature={"h_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "E": "constexpr", "T": "constexpr", "F": "constexpr",
               "F2": "constexpr", "ALPHA": "constexpr", "LIMIT": "constexpr"},
    constexprs={"E": E, "T": T, "F": F, "F2": F2, "ALPHA": ALPHA,
                "LIMIT": LIMIT},
    args=[Arg("h", IN, "float32", (E, T, F2)),
          Arg("b", IN, "float32", (E, F2)),
          Arg("o", OUT, "float32", (E, T, F))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
