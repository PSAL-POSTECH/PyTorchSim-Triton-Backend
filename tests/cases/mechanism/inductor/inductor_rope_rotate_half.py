"""RoPE's `rotate_half` as Inductor writes it: a 256-wide tile split by `% 128`,
with the two halves selected by masks computed from the index itself.

This kernel pins TWO defects, both of which it hit in that order, and both of
which are about the same thing -- a rewrite that moved only half of what it had
to move. It is the smallest form of Qwen's kernel 4 that reaches either.

  1  A PLAN ARM THAT WAS ABANDONED LEFT ITS MARKS (absorb_layout_ops).
     A plan records which values it has already scheduled a retype for, so that a
     value reached twice is retyped once. `_plan_iteration` is TRIED and fallen
     through from -- if it cannot take the move, `_plan_def_elementwise` runs
     instead -- and it marked values on its way to failing. The arm that ran next
     then read those marks as its own promise and emitted nothing for them, so
     the maps moved to a 2-D iteration space while the operands stayed rank 1:

         'linalg.generic' op expected operand #0 rank (1) to match the result
         rank of indexing_map (2)

     `_attempted` is the fix: an arm plans on a snapshot, committed only if it
     answers.

  2  A SUB-WORD TILE COULD BE FOLDED INTO ONE BANK BUT NOT SPREAD OUT OF ONE
     (fit_layout_to_lanes). A push carries 32 bits, so a sub-word element crosses
     AS a word or not at all -- and the widen/narrow pair that says so was
     attached to the FOLD alone. The masks here bufferize to i8, so they folded
     into one bank with their f32 siblings and then could not come back out:

         one operand of this elementwise op is in a single bank (ONE_LANE) while
         another is banked across the lanes on iteration dim 1 of extent 128

     The pair now lives in `_emit_transpose` itself, which is the only place a
     transpose is minted, so neither direction can be the one that forgot.

WHY THE THIRD LOAD, THE ONE INDEXED BY x3. Without it the move is the same on
every operand and `_plan_iteration` succeeds, so nothing is ever abandoned and
the kernel passes on both trees. `in_ptr2 + x3` rides the OTHER half of the
split, and that is the operand whose own move the arm cannot take.

AND WHY ITS NINE RELAYOUTS ARE STILL DRAM ROUND TRIPS. The pair here is a 256
row against the (2, 128) it is banked as, and a round of the unit owns a RUN of
lanes: out is 128 rounds of two lanes, but home every source row feeds lanes
`d` and `d + 128` -- two lanes 128 apart, which is neither a run nor a straight
line of the pattern. A written operand needs both ways, so both go through DRAM.

torch equivalent: see reference -- the two masked halves of a rotate_half, each
scaled by an rsqrt of a per-row mean, recombined by the same mask.

Qwen's RoPE kernel at its smallest: pins a plan arm that is abandoned leaving no marks, and a sub-word tile leaving one bank.

LAYOUT PIN. `expect_ir` holds `tensor<2x128xf32>` in 03a-ttshared.mlir at 0; what that pins is the split load keeping its layout (triton_shared: the split pointer and the load take the reshape's inferred encoding); without it the 2x128 tile reaches p02 bare, 2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

XB, W, H, R = 256, 128, 64, 2


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _rs(v):
        return libdevice.rsqrt(v / 128.0 + 1e-6)

    @triton.jit
    def inductor_rope_rotate_half_kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr1,
                                         XB: tl.constexpr, W: tl.constexpr,
                                         H: tl.constexpr):
        xindex = tl.arange(0, XB)
        x0 = xindex % W
        x3 = xindex // W
        th = tl.full([1], H, tl.int64)
        lo = x0 < th
        hi = x0 >= th
        a0 = tl.load(in_ptr0 + (H + x0), lo, other=0.0)
        a1 = tl.load(in_ptr1 + (H + W * x3 + x0), lo, other=0.0)
        a2 = tl.load(in_ptr2 + x3, lo, other=0.0)
        av = tl.where(lo, -(a0 * (a1 * _rs(a2))), 0.0)
        b0 = tl.load(in_ptr0 + ((-H) + x0), hi, other=0.0)
        b1 = tl.load(in_ptr1 + (W * x3 + ((-H) + x0)), hi, other=0.0)
        b2 = tl.load(in_ptr2 + x3, hi, other=0.0)
        bv = tl.where(hi, b0 * (b1 * _rs(b2)), 0.0)
        c2 = tl.load(in_ptr2 + x3)
        tl.store(out_ptr1 + xindex, tl.where(lo, av, bv) + _rs(c2))

    return inductor_rope_rotate_half_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in_ptr0": torch.randn(W, generator=g, dtype=torch.float32),
            "in_ptr1": torch.randn(XB, generator=g, dtype=torch.float32),
            "in_ptr2": torch.rand(R, generator=g, dtype=torch.float32) + 1.0}


def reference(inputs):
    import torch
    p0, p1, p2 = inputs["in_ptr0"], inputs["in_ptr1"], inputs["in_ptr2"]
    i = torch.arange(XB)
    x0, x3 = i % W, i // W
    lo, hi = x0 < H, x0 >= H
    z = torch.zeros(XB)

    def rs(v):
        return torch.rsqrt(v / 128.0 + 1e-6)

    a0 = torch.where(lo, p0[(H + x0).clamp(0, W - 1)], z)
    a1 = torch.where(lo, p1[(H + W * x3 + x0).clamp(0, XB - 1)], z)
    a2 = torch.where(lo, p2[x3], z)
    b0 = torch.where(hi, p0[((-H) + x0).clamp(0, W - 1)], z)
    b1 = torch.where(hi, p1[(W * x3 + ((-H) + x0)).clamp(0, XB - 1)], z)
    b2 = torch.where(hi, p2[x3], z)
    av = torch.where(lo, -(a0 * (a1 * rs(a2))), z)
    bv = torch.where(hi, b0 * (b1 * rs(b2)), z)
    return {"out_ptr1": torch.where(lo, av, bv) + rs(p2[x3])}


SPEC = KernelSpec(
    name="inductor_rope_rotate_half_kernel", kernel=kernel,
    signature={"in_ptr0": "*fp32", "in_ptr1": "*fp32", "in_ptr2": "*fp32",
               "out_ptr1": "*fp32", "XB": "constexpr", "W": "constexpr",
               "H": "constexpr"},
    constexprs={"XB": XB, "W": W, "H": H},
    args=[Arg("in_ptr0", IN, "float32", (W,)),
          Arg("in_ptr1", IN, "float32", (XB,)),
          Arg("in_ptr2", IN, "float32", (R,)),
          Arg("out_ptr1", OUT, "float32", (XB,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"03a-ttshared.mlir": {"tensor<2x128xf32>": 0}},
)
