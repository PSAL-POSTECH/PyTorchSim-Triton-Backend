"""CLIP's head -- L2-normalise both towers, matmul them, scale by
`logit_scale.exp()`. The whole contrastive objective is these four lines.

    ie = img / ||img||_2            per row, over the embedding dim
    te = txt / ||txt||_2
    logits_per_text  = te @ ie.T * logit_scale.exp()
    logits_per_image = logits_per_text.T

L2 NORMALISATION IS NOT A NORM LAYER, AND THE DIFFERENCE IS THE POINT. Every
other normalisation in this suite -- `layernorm.py`, `rmsnorm.py`, `qk_norm.py`
-- ends in a LEARNED per-feature affine, and LayerNorm subtracts a mean first.
This one does neither: no mean, no gamma, no beta, no epsilon in the shipped
code. It is `x / sqrt(sum(x*x))`, a projection onto the unit sphere, and the
consequence is that the matmul that follows computes a COSINE. That is what
makes the similarity comparable across pairs at all -- an unnormalised dot
product would let the model win the contrastive loss by growing the norm of a
few embeddings instead of by aligning directions.

  Structurally it is RMSNorm with the `mean` replaced by a `sum` and the weight
  deleted, which sounds like nothing and is the difference between a layer with
  parameters and a geometric constraint. A backend that pattern-matched
  "reduction, rsqrt, broadcast, multiply" as normalisation would produce the
  right number here; the file exists because the CONSUMER is different -- the
  normalised tile goes straight into `tl.dot`, so one buffer carries the
  reduction's "not the lane axis" and the systolic array's "this axis", the
  collision `qk_norm.py` documents, with the reduction now being a plain sum of
  squares and both operands of the matmul carrying it.

WHY THE TEMPERATURE IS EXP OF A LOADED SCALAR. `logit_scale` is a single learned
parameter stored as a LOGARITHM, because the loss wants a positive temperature
and optimising in log space keeps it positive without a constraint. So the
kernel's scale is `exp` of a rank-0 value read from DRAM: a scalar load, a
transcendental on it, and then a broadcast of one number over an entire tile --
`fused_moe_expert_mlp.py` loads a scalar routing weight the same way, but there
it multiplies rows, and here it multiplies the matmul's full output. OpenAI
clamps the parameter to log(100) during training; the clamp lives on the
parameter, not on this expression.

BOTH ORIENTATIONS ARE OUTPUTS. The contrastive loss is a cross-entropy along
each axis of the same matrix -- image-to-text and text-to-image -- so the
reference implementation returns `logits_per_text` and its transpose, and both
are stored here. The two use different row strides over the same values, which
is the cheapest possible check that the matmul's output layout is what the
backend thinks it is.

SHAPES. 32 images against 16 texts, deliberately unequal: with a square matrix a
transposed store is indistinguishable from a plain one.

torch equivalent:
    ie = img / img.norm(p=2, dim=-1, keepdim=True)
    te = txt / txt.norm(p=2, dim=-1, keepdim=True)
    lpt = te @ ie.t() * ls.exp();  lpi = lpt.t()

GREEN SINCE 2026-09-04. vectorize refused this with "operands of one
elementwise op are banked on different iteration dims": one buffer a matmul pins
on one axis sits in an all-parallel op beside buffers that follow the op onto the
other. select_lane_axis owned that decision and did not know it -- its
`_is_one_iteration_dim` reported only the disagreements a widen could already
fix, so the ladder never reached step 4 and the refusal fell out a pass later.

The pinned operand here is the op's `outs`, and a relayout mints a new value
while the op already produces that one. So the cut goes on the CONSUMER instead:
the matmul that pins it reads a copy banked its way, and the producing op is free
to follow its own operands. Same trip, no new rewrite. Costs 1 round trip.

CLIP: L2 (not RMS, not LayerNorm -- no mean, no affine) on both towers feeding a matmul, times exp of a scalar loaded from DRAM, stored in both orientations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NI = 32        # images in the batch
NT = 16        # texts -- not NI, so the transposed store is a real check
D = 64         # projection dim


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def clip_contrastive_logits_kernel(img_ptr, txt_ptr, ls_ptr,
                                       lpt_ptr, lpi_ptr, NI: tl.constexpr,
                                       NT: tl.constexpr, D: tl.constexpr):
        ri = tl.arange(0, NI)
        rt = tl.arange(0, NT)
        d = tl.arange(0, D)

        img = tl.load(img_ptr + ri[:, None] * D + d[None, :])
        txt = tl.load(txt_ptr + rt[:, None] * D + d[None, :])

        # L2, NOT RMS: a SUM of squares, no mean, no eps, no learned weight.
        img = img / tl.sqrt(tl.sum(img * img, axis=1))[:, None]
        txt = txt / tl.sqrt(tl.sum(txt * txt, axis=1))[:, None]

        # The temperature: one learned number, stored as its log.
        scale = tl.exp(tl.load(ls_ptr))

        lpt = tl.dot(txt, tl.trans(img)) * scale          # [NT, NI]
        tl.store(lpt_ptr + rt[:, None] * NI + ri[None, :], lpt)
        tl.store(lpi_ptr + ri[:, None] * NT + rt[None, :], tl.trans(lpt))

    return clip_contrastive_logits_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"img": torch.randn(NI * D, generator=g, dtype=torch.float32),
            "txt": torch.randn(NT * D, generator=g, dtype=torch.float32),
            # log(1/0.07), CLIP's initialisation and roughly where it stays.
            "ls": torch.full((1,), 2.6592, dtype=torch.float32)}


def reference(inputs):
    import torch
    img = inputs["img"].reshape(NI, D)
    txt = inputs["txt"].reshape(NT, D)
    ie = img / img.norm(p=2, dim=-1, keepdim=True)
    te = txt / txt.norm(p=2, dim=-1, keepdim=True)
    lpt = te @ ie.t() * inputs["ls"][0].exp()
    return {"lpt": lpt.reshape(-1).contiguous(),
            "lpi": lpt.t().reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="clip_contrastive_logits_kernel",
    kernel=kernel,
    signature={"img_ptr": "*fp32", "txt_ptr": "*fp32", "ls_ptr": "*fp32",
               "lpt_ptr": "*fp32", "lpi_ptr": "*fp32", "NI": "constexpr",
               "NT": "constexpr", "D": "constexpr"},
    constexprs={"NI": NI, "NT": NT, "D": D},
    args=[Arg("img", IN, "float32", (NI * D,)),
          Arg("txt", IN, "float32", (NT * D,)),
          Arg("ls", IN, "float32", (1,)),
          Arg("lpt", OUT, "float32", (NT * NI,)),
          Arg("lpi", OUT, "float32", (NI * NT,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
