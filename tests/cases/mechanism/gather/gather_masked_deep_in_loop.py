"""a gathered MVIN filling a tile DEEPER than the lane count, per lane.

PASSES. It did not when it was written, and what it pins is the fix.

WHAT IT USED TO DO, and what a regression here would look like: a lane was
filled only `lanes` elements deep and every further stride re-read that same
run, so the gathered tile came back with its first lane-count columns repeated.
The limit was the LANE COUNT -- not a constant, and not "more than one element
per lane", which is what this file claimed at 82871d0 before the machine was
moved underneath it:

    lanes 128    R0_BLOCK 128  PASS      R0_BLOCK 256  FAIL
    lanes  64    R0_BLOCK  64  PASS      R0_BLOCK 128  FAIL

and the tile the kernel gathered, recovered from its own outputs, repeated
with exactly that period:

    lanes 128    cols 128-255 == cols   0-127,  384-511 == 256-383,  ...
    lanes  64    cols  64-127 == cols   0- 63,  192-255 == 128-191,  ...

Nothing was refused and nothing warned -- it compiled clean and returned wrong
numbers, the failure rule 12 exists for. FIXED by develop-select-grid 2007619
("Take the depth from the vector too, and the wide-tile gather comes right"),
merged here; measured on this tree s 7.62939e-06, q 0.00012207, o 7.62939e-06.

WHY THIS IS THE INTERESTING NUMBER. `R0_BLOCK` is PER LANE and is deliberately
NOT the lane count: PyTorchSim sizes it from the scratchpad budget
(`kernel_spec.fixed_config_for`, and its own comment says the reduced axis has
to live INSIDE one lane), so it is free to exceed `lanes` and for GPT-2 it does
-- 768 -> 1024 -> 512 by budget, then 256 after one spad-overflow recompile.
The mm path does NOT have that freedom: it caps its tiles AT the lane count,
and `tests/ops/attention/test_gqa.py` is what that cap holds up (lift it and
gqa comes back 0.0954 off, measured on develop-select-grid). That cap is this
defect being papered over from the other side. The reduction path has no cap,
so it walks straight into it.

REDUCED FROM GPT-2 (tests/models/GPT2/test_gpt2.py --preset small), kernel 0
`triton_npu_fused_add_arange_embedding_native_layer_norm_unsqueeze_0` -- the
embedding gather plus the first LayerNorm, the first buffer the model computes
and the first one wrong. `--preset tiny` (n_embd 128) passes end to end at
4.77e-07 because its blocks never exceed the lane count. BERT stops in the same
place at its own small preset.

FIVE INGREDIENTS, measured by leave-one-out on the BROKEN tree -- every variant
passed there, so all five were needed to reach the defect and all five are kept
so this kernel still covers the whole path:

    1  a direct load instead of the gather            PASSES
    2  masked with xmask only, without r0_mask        PASSES
    3  R0_BLOCK at the lane count instead of over it  PASSES
    4  N 256, so the loop body runs once              PASSES
    5  the second gather loop deleted                 PASSES

  (5) is why this file has a second loop that looks redundant: ONE gathered
  transfer is correct, two sharing an index buffer are not -- the shape
  develop-e2e-bert filed at 12b2256 and develop-select-grid's e62c84c fixed for
  the bmm. This is the case that fix does not reach.

RULED OUT, so nobody re-walks them. Each was written as a kernel and run:

    Welford's three-way carry               NOT the ingredient -- the same
                                            kernel with Welford and with two
                                            plain sums fail IDENTICALLY
    a general pair combiner over loop
      partials (reduce_general_over_loop_partials at R 64/128/256)   passes
    a looped reduce over the lane count with plain sums, no gather   passes
    gather + mask + depth + store, no reduction at all               passes

THE SIBLING REPRO, and it is the same defect from the other side:
`tile/tile_bias_row_deeper_than_one_per_lane.py` (develop-select-grid 45f4542)
pins an addmm epilogue's REPLICATING bias row -- a [BM, BN] tile whose DRAM row
stride is 0 -- breaking at BN 256 and 512 with BN 128 clean. Different transfer,
same threshold. Neither subsumes the other: that one is unmasked, unlooped and
not gathered. Fix either, run both.

A gathered MVIN fills only LANE-COUNT elements per lane and then repeats that run: measured at lanes 128 (256 fails, 128 passes) and at lanes 64 (128 fails, 64 passes), with the replication period tracking the lane count both times. Reduced from GPT-2 --preset small kernel 0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS = 32           # xnumel; masked under XBLOCK, as Inductor leaves it
XBLOCK = 128        # the lane count: the ROW axis is the lane axis
N = 768             # three blocks, so the loop runs more than twice
R0_BLOCK = 256      # 2x the lanes -- the depth that matters
VOCAB = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_masked_deep_in_loop_kernel(idx_ptr, tab_ptr, s_ptr, q_ptr, o_ptr,
                                          ROWS: tl.constexpr,
                                          VOCAB: tl.constexpr,
                                          N: tl.constexpr,
                                          R0_BLOCK: tl.constexpr,
                                          XBLOCK: tl.constexpr):
        x = tl.arange(0, XBLOCK)[:, None]
        xmask = x < ROWS
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        row = tl.load(idx_ptr + x, xmask)

        acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        acc2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        for r0_offset in range(0, N, R0_BLOCK):
            r0 = r0_offset + r0_base
            r0_mask = r0 < N
            # THE GATHER. `r0_mask & xmask` names BOTH axes -- with xmask alone
            # this kernel passes.
            v = tl.load(tab_ptr + (r0 + N * row), r0_mask & xmask, other=0.0)
            acc += v
            acc2 += v * v

        total = tl.sum(acc, 1)[:, None]
        tl.store(s_ptr + x, total, xmask)
        tl.store(q_ptr + x, tl.sum(acc2, 1)[:, None], xmask)

        # THE SECOND GATHER. Same table, same index tile, a second loop -- two
        # gathered transfers sharing one index buffer. Delete this loop and the
        # kernel passes.
        for r0_offset in range(0, N, R0_BLOCK):
            r0 = r0_offset + r0_base
            r0_mask = r0 < N
            v = tl.load(tab_ptr + (r0 + N * row), r0_mask & xmask, other=0.0)
            tl.store(o_ptr + (r0 + N * x), v - total, xmask)

    return gather_masked_deep_in_loop_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"idx": torch.randint(0, VOCAB, (ROWS,), generator=g, dtype=torch.int64),
            "tab": torch.randn(VOCAB, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["tab"][inputs["idx"]]
    return {"s": x.sum(1), "q": (x * x).sum(1),
            "o": x - x.sum(1, keepdim=True)}


SPEC = KernelSpec(
    name="gather_masked_deep_in_loop_kernel",
    kernel=kernel,
    signature={"idx_ptr": "*i64", "tab_ptr": "*fp32", "s_ptr": "*fp32",
               "q_ptr": "*fp32", "o_ptr": "*fp32", "ROWS": "constexpr", "VOCAB": "constexpr",
               "N": "constexpr", "R0_BLOCK": "constexpr", "XBLOCK": "constexpr"},
    constexprs={"ROWS": ROWS, "VOCAB": VOCAB, "N": N,
                "R0_BLOCK": R0_BLOCK, "XBLOCK": XBLOCK},
    args=[Arg("idx", IN, "int64", (ROWS,)),
          Arg("tab", IN, "float32", (VOCAB, N)),
          Arg("s", OUT, "float32", (ROWS,)),
          Arg("q", OUT, "float32", (ROWS,)),
          Arg("o", OUT, "float32", (ROWS, N))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
