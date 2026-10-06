"""A unit axis dropped from the ONE map that still named it, and the loop bound
that goes with it.

    op      linalg.generic, ins(%14) outs(%13), BOTH maps the identity over
            tensor<8x1x32xf32> -- a single-input elementwise square
    move    drops the unit d1
    after   maps = [(d0,d1,d2) -> (d0,d2), (d0,d1,d2) -> (d0,d2)]

            p06_absorb_layout_ops._Fatal: indexing map cannot take the move

WHY IT REFUSES, measured by instrumenting the two `return None`s in
`_restated_maps` rather than read off the shape:

    PROBE: dim lost its bare mention; named=[0, 2] n_dims=3

Not the first return -- the restatement itself succeeds. The second: with both
maps restated there is no bare `d1` left anywhere in the op, so linalg can no
longer infer that loop's bound from the operands and would raise "expected the
shape-to-loops map to be non-null" at verify. `_restated_maps` says exactly this
case in its own docstring: "Every generic in the suite whose unit axis is
dropped has another operand naming the same dim -- causal_conv1d's
(d0,d1,d2)->(d0,d1) weight beside the (d0,d1,d2)->(d0,d1,d2) tile -- and the one
that does not is the one this refuses."

AND THE REASON DOES NOT HOLD FOR A UNIT AXIS. The bound it cannot infer is 1.
That argument already landed one function over, in "A unit axis that goes away
takes no name with it" (efaf26c), which taught `_restate_names` the same thing
in the same file: "a unit axis is not an axis ... it holds one element, so it
steps nothing". A name about a unit axis is dropped rather than refused there;
a map result about one is still refused here.

WHAT IT TAKES, each measured by removing it and watching the kernel go green:

    two loops           a reduction loop and a second loop that CONSUMES its
                        result. With the second loop deleted and the sum stored
                        instead, it compiles.
    the per-row load    `p_ptr + x0`, the `(d0,d1)` operand. Removing it alone
                        compiles -- it is what puts the op group in reach of a
                        move at all.
    the r-only load     `q_ptr + r`, broadcast over x in the SECOND loop.
                        Removing it alone compiles.

    NOT the gate        the scalar `tl.load(g + 0)` and its tanh, which the
                        Mllama kernel this came from carries and which its name
                        (`..._select_tanh_24`) makes look load-bearing. It is
                        not: removing both keeps the refusal.

WHERE IT CAME FROM. Llama 3.2 Vision, PyTorchSim
tests/models/Llama/test_mllama.py --part text -- MllamaTextModel's
cross-attention decoder layer, whose norm sits between exactly these two loops.
Reduced from that kernel by deletion, at 32x1024; 8x32 keeps it.

WHAT IS ACTUALLY WRONG, one step further in than the message. At the refusal
the op's operands are BOTH rank 2 while its first map is rank 3:

    maps    [(d0,d1,d2) -> (d0,d1,d2),  (d0,d1,d2) -> (d0,d2)]
    shapes  [8, 32]                      [8, 32]

so the unit axis is already gone from the TYPES and still written in the map.
`_dim_extents` reads extents off map results that are bare dims, and d1's
operand axis does not exist, so it comes back `[8, None, 32]` -- d1 has no
extent anywhere. The refusal is then correct on its own terms: it cannot know
the bound is 1, because nothing in the op still says so.

WHICH MOVES THE QUESTION UP, to whoever took the axis out of the operand type
without taking it out of the map. That is a rule-18 half-rewrite -- shape moved,
the statements hanging off the axis did not -- and it is not this line's to fix.

WHAT FIXED IT, and it is one blindness showing up in two places. Dropping the
LAST bare mention of a dim is exactly when `_dim_extents` can no longer answer
for it, so the refusal here and the cleanup below it fail together:

    _restated_maps      refuses, because it cannot see the bound is 1
    _unit_iterations    would not drop the dim either, for the same reason

Relaxing only the first leaves `iterator_types` three long against maps naming
two dims. That does not verify -- "invalid indexing maps are non-invertible:
((d0,d1,d2) -> (d0,d2,d0,d2))" -- and one-shot-bufferize SEGFAULTS on it rather
than saying so. Measured, and it is the half-rewrite rule 4 names, in the
version that does not even verify.

Both now read the extent off the MOVE instead: `_Move` defines a gap as an axis
of extent 1 going away, which is true whatever has already been applied to the
operands. `_gap_dims` is what the two share.

TWO EARLIER REPAIRS, WRITTEN AND REVERTED, kept so they are not written again:
reading the extent off the operands is inert (it answers None, not 1, because
the operand has already moved); relaxing the refusal alone is the segfault
above.

MEASURED at 8x32: 4.76837e-07 against the torch reference on spike, every stage
green. Before the change: "_Fatal: indexing map cannot take the move".

WHAT WOULD BREAK IT. Reading the extent off the OPERAND rather than off the
move. The dim is unit in the iteration space; the operand whose map named it
may already be rank 2 by the time this runs, which is the whole reason the
question is hard.

torch equivalent: o[i,j] = q[j] * v[i,j] * rsqrt(mean_j(v[i,j]**2) + eps),
                  v = x + p[i]*y[i,j]

The reproducer for Llama 3.2 Vision's cross-attention layer (PyTorchSim tests/models/Llama/test_mllama.py --part text). Reduced from that kernel by deletion: two loops, the (d0,d1) per-row load and the r-only load in the second loop are each necessary; the scalar gate and its tanh, which the kernel's name advertises, are not.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, R = 8, 32
EPS = 1e-5


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def unit_axis_last_bare_mention(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
        xnumel = 8
        r0_numel = 32
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = xindex
        tmp1 = tl.load(in_ptr1 + (0))
        tmp2 = tl.broadcast_to(tmp1, [1, 1])
        tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
        _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
            tmp5 = tl.load(in_ptr3 + (r0_1 + 32*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
            tmp7 = tmp4 * tmp5
            tmp8 = tmp0 + tmp7
            tmp9 = tmp8 * tmp8
            tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
            tmp12 = _tmp11 + tmp10
            _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tmp11 = tl.sum(_tmp11, 1)[:, None]
        tmp15 = tl.load(in_ptr1 + (0))
        tmp16 = tl.broadcast_to(tmp15, [1, 1])
        for r0_offset in range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_1 = r0_index
            tmp13 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
            tmp14 = tl.load(in_ptr0 + (r0_1 + 32*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
            tmp18 = tl.load(in_ptr3 + (r0_1 + 32*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
            tmp20 = tmp4 * tmp18
            tmp21 = tmp14 + tmp20
            tmp22 = 32.0
            tmp23 = (tmp11 / tmp22)
            tmp24 = 1e-05
            tmp25 = tmp23 + tmp24
            tmp26 = libdevice.rsqrt(tmp25)
            tmp27 = tmp21 * tmp26
            tmp28 = tmp13 * tmp27
            tl.store(out_ptr1 + (r0_1 + 32*x0), tmp28, r0_mask & xmask)
    return unit_axis_last_bare_mention


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in_ptr0": torch.randn(X * R, generator=g, dtype=torch.float32),
            "in_ptr1": torch.randn(1, generator=g, dtype=torch.float32),
            "in_ptr2": torch.randn(X, generator=g, dtype=torch.float32),
            "in_ptr3": torch.randn(X * R, generator=g, dtype=torch.float32),
            "in_ptr4": torch.randn(R, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["in_ptr0"].reshape(X, R)
    p = inputs["in_ptr2"].reshape(X, 1)
    y = inputs["in_ptr3"].reshape(X, R)
    q = inputs["in_ptr4"].reshape(1, R)
    v = x + p * y
    s = (v * v).sum(dim=1, keepdim=True) / R
    return {"out_ptr1": (q * (v * torch.rsqrt(s + EPS))).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="unit_axis_last_bare_mention", kernel=kernel,
    signature={"in_ptr0": "*fp32", "in_ptr1": "*fp32", "in_ptr2": "*fp32",
               "in_ptr3": "*fp32", "in_ptr4": "*fp32", "out_ptr1": "*fp32",
               "xnumel": "i32", "r0_numel": "i32",
               "XBLOCK": "constexpr", "R0_BLOCK": "constexpr"},
    constexprs={"XBLOCK": X, "R0_BLOCK": R},
    args=[Arg("in_ptr0", IN, "float32", (X * R,)),
          Arg("in_ptr1", IN, "float32", (1,)),
          Arg("in_ptr2", IN, "float32", (X,)),
          Arg("in_ptr3", IN, "float32", (X * R,)),
          Arg("in_ptr4", IN, "float32", (R,)),
          Arg("out_ptr1", OUT, "float32", (X * R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("xnumel", "int32_t"), ("r0_numel", "int32_t")],
           "scalar_values": {"xnumel": X, "r0_numel": R}},
    rtol=1e-4, atol=1e-4,
)
