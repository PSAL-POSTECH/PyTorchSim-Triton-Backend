"""Mamba-2 / Qwen3-Next `RMSNormGated` -- a SiLU gate multiplied INTO the norm,
normalised per head rather than per token.

    g = x * silu(z)                                  # the gate, elementwise
    r = rsqrt(mean(g*g over the HEAD's channels) + eps)
    y = g * r * w                                    # w is per channel

WHY THE GATE IS INSIDE THE NORM. Mamba-2's block ends with a gated output branch:
the SSM produces `x` and a parallel projection produces `z`, and the layer output
is `norm(x * silu(z)) @ W_out`. Putting the gate in front of the norm and not
after it is a modelling decision, not a fusion: gating first means the norm sees
the gated magnitudes and rescales them, so a head the gate has mostly closed is
renormalised back up instead of vanishing. `norm_before_gate=False` in
mamba-ssm's `RMSNormGated`, and it is the setting every released Mamba-2
checkpoint uses. Qwen3-Next carries the same module verbatim.

WHY IT IS ONE KERNEL. `z` is the same size as `x`, which is d_inner -- the widest
tensor in the block. Unfused this is a read of x, a read of z, a write of g, a
read of g back for the reduction, and a write of y. Fused it is two reads and one
write, and the gated value never leaves registers.

WHAT IS STRUCTURALLY DIFFERENT FROM fused_add_rmsnorm.py, THE NEAREST KERNEL.
Two things, and both change the tile.

  * THE COMBINE BEFORE THE REDUCTION IS MULTIPLICATIVE, NOT ADDITIVE. That
    kernel's pre-reduction value is `x + residual` and it is ALSO an output --
    the next residual. Here the pre-reduction value is `x * silu(z)` and it is
    consumed and dropped; nothing downstream wants it. So the pressure is not
    "hold a tile live across a reduction to store it", it is a longer
    elementwise chain (sigmoid, two multiplies) feeding the reduction, with a
    second full-width operand streaming in beside the first.

  * THE REDUCTION IS PER GROUP, NOT PER ROW. That kernel reduces a whole row of
    H channels. This one splits the row into HD heads of DH channels and reduces
    EACH SEPARATELY -- it is GroupNorm's grouping wearing RMSNorm's arithmetic,
    which is what mamba-ssm's `group_size` argument does and what makes the norm
    match the multi-head structure of the SSM above it. A partial-row reduction
    is the awkward case: the reduced extent is DH, not the row length, so the
    natural tile is one that puts (token, head) on the lane axis and DH in-lane.
    That is the layout here, and it costs a MODULO -- the per-channel weight is
    indexed by `row % HD`, because the weight is laid out per hidden channel
    while the tile is laid out per group. Nothing else in the suite derives a
    load offset from a remainder.

THE RECIPROCAL STDDEV IS AN OUTPUT. `y` folds gate, norm and weight together, so
a mismatch there says nothing about which of the three moved. `r` isolates the
reduction: it depends on the gate and on the grouping and on nothing else, so a
`y` that is wrong while `r` is right is a weight-indexing fault, and the reverse
is a grouping fault.

eps is inside the sqrt, as in the reference module. There is no mean to subtract
-- RMS, not LayerNorm -- so the dependency through the reduction is one deep.

Mamba-2 RMSNormGated: a SiLU gate folded in ahead of a PER-HEAD reduction, with the weight reached through a remainder. fused_add_rmsnorm.py is additive and reduces the whole row.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, HD, DH = 8, 4, 16            # tokens, heads (groups), channels per head
TH = T * HD                     # one reduction group per lane
HID = HD * DH                   # hidden width the weight is laid out over
EPS = 1e-5


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rmsnorm_gated_kernel(x_ptr, z_ptr, w_ptr, y_ptr, r_ptr,
                             TH: tl.constexpr, HD: tl.constexpr,
                             DH: tl.constexpr, EPS: tl.constexpr):
        # Lane axis is (token, head) flattened; in-lane axis is the group.
        th = tl.arange(0, TH)[:, None]
        c = tl.arange(0, DH)[None, :]
        off = th * DH + c                       # contiguous: [T, HD, DH]

        z = tl.load(z_ptr + off)
        # THE GATE, multiplied in before anything is reduced.
        g = tl.load(x_ptr + off) * (z * tl.sigmoid(z))

        # PER-GROUP reduction: the extent is DH, not the hidden width.
        r = 1.0 / tl.sqrt(tl.sum(g * g, axis=1) / DH + EPS)
        tl.store(r_ptr + tl.arange(0, TH), r)

        # The weight is per HIDDEN channel; the tile is per group. The head index
        # is the row's remainder, which is where the two layouts are reconciled.
        w = tl.load(w_ptr + (th % HD) * DH + c)
        tl.store(y_ptr + off, g * r[:, None] * w)

    return rmsnorm_gated_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(TH * DH, generator=g, dtype=torch.float32),
            "z": torch.randn(TH * DH, generator=g, dtype=torch.float32),
            "w": torch.randn(HID, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, HD, DH)
    z = inputs["z"].reshape(T, HD, DH)
    g = x * torch.nn.functional.silu(z)
    r = (g.pow(2).mean(dim=2) + EPS).rsqrt()                 # [T, HD]
    y = g * r[:, :, None] * inputs["w"].reshape(HD, DH)[None, :, :]
    return {"y": y.reshape(-1).contiguous(), "r": r.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="rmsnorm_gated_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "z_ptr": "*fp32", "w_ptr": "*fp32",
               "y_ptr": "*fp32", "r_ptr": "*fp32", "TH": "constexpr",
               "HD": "constexpr", "DH": "constexpr", "EPS": "constexpr"},
    constexprs={"TH": TH, "HD": HD, "DH": DH, "EPS": EPS},
    args=[Arg("x", IN, "float32", (TH * DH,)),
          Arg("z", IN, "float32", (TH * DH,)),
          Arg("w", IN, "float32", (HID,)),
          Arg("y", OUT, "float32", (TH * DH,)),
          Arg("r", OUT, "float32", (TH,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
