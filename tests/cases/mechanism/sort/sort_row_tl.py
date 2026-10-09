"""`tl.sort` on a row -- triton's own bitonic network, and the one Inductor emits.

torch equivalent: x.sort(dim=1).values

WHY THIS ONE MATTERS MOST OF THE THREE
  `torch.sort` and `torch.topk` do not reach a backend as a sort op. Inductor
  lowers them to `triton_helpers.sort_with_index`, which is a bitonic network
  written in triton, and `tl.sort` is the same algorithm exposed as a builtin.
  So this kernel is the witness for the shape the Triton codegen route will
  actually hand over -- sort_bitonic.py spells the network out by hand to say
  what that expands to, but THIS is what arrives.

WHAT A BITONIC NETWORK NEEDS FROM THE BACKEND
  log2(N) * (log2(N)+1) / 2 compare-exchange stages, and every stage moves data
  ALONG THE SORTED AXIS by a power-of-two stride. That is the part with no
  reduction in it: a reduction collapses an axis and this permutes it. The
  scratchpad is lane-banked on dim 0, so the whole network has to stay inside a
  lane -- exactly like reduce_row_max.py, and for the same reason.

N IS A POWER OF TWO because a bitonic network is only defined on one. 16 keeps
the network at 10 stages, small enough to read in 11-kernel.s.

WHAT STOPS IT TODAY -- A UNIT ROW NOBODY ARRANGED TO REPLICATE. It used to reach
the suite as `KILLED by signal 6 (SIGABRT) -- no diagnostic was printed`. That
was two faults stacked: vectorize's `_mask_tail` built its keep-mask from a
list of PYTHON BOOLS, which DenseElementsAttr has no path for (`std::bad_cast`),
and the same pass drove `_nest` with a bare `__enter__` so the raise abandoned
its insertion points -- masking the real exception and aborting the interpreter
at shutdown. Both are fixed, and what the kernel actually needs is now readable:

    NotVectorisable: operand [1, ..., 2, 1] broadcasts along dim 0 into a tile
    [2, ..., 2], and it was staged one element per lane -- every lane needs the
    whole row, which takes a replicating transfer this operand did not get

fit_layout_to_lanes is where that is decided -- `_stage_one_map_row` ranks the
four stagings a row can get, and `_clone_loaded_row_wide` is the free one:
a second transfer of the same DRAM with a zero stride on the broadcast
axis. It needs a DRAM address to re-read. Here they are `tl.sort`'s own stage masks: the rank-11 form triton emits
for a 2048-wide bitonic network, whose partner and direction rows are
computed from `linalg.index`.
So the row is staged into ONE lane and vectorize refuses rather than read
another lane's bank. `_replicate_computed_row` is the answer that does not need
one -- a DRAM round trip with a zero stride -- and reaching it from here is the
fix; it is the same fix for
block_diag_attn_mask, paged_attention and top_p_sampling.

GREEN SINCE 2026-09-04. `tl.sort` reshapes its tile into log2(n) axes of extent 2
so a compare-exchange at stride 2^k can NAME its partner, and the reassociation
says which axes are which -- for a [128, 16] tile that is `[[0..6], [7..10]]`,
seven spelling the rows and four the columns. absorb_layout could only take a
reshape whole: forward (readers see [128, 16]) is refused for a real reason, since
merging the column run takes away the ability to name axis 3 alone, which is what
the network is FOR. So backward won, the tile stayed rank 11 to the end,
select_lane_axis had eleven interchangeable axes of 2 to choose between, and the
DMA descriptor -- which holds four dims -- peeled the lane axis into an
`affine.for`, where an outer loop iterates the axis the DMA is meant to spread.

The measurement that unlocked it: `_iteration_takes` refuses the merge 21 times
and EVERY ONE of the 21 is run 1, the columns. Run 0, the rows, is refused zero
times. "The columns cannot merge" was silently deciding "the rows cannot either".
A reshape may now merge ONE of its runs, the tile arrives as [128, 2, 2, 2, 2],
and the DMA folds the four contiguous binary axes back into 16 -- never across
the lane axis, which is the clause that made the same fold wrong on its own.

CORE DUMPS, and the crash is worth more than the sort. Stage 3 refuses this IR, and on the way out vectorize._nest's cleanup raises `Unbalanced InsertionPoint enter/exit`, which leaves the context stack unbalanced so stage_transform then reports `Unbalanced Location enter/exit`, and at shutdown the dangling MLIR context hits `PyThreadState_Get ... the GIL is released` and aborts (exit 134). The original refusal never gets printed. That is the same SIGABRT four PyTorchSim Triton-route tests hit -- attention_fusion, transformer_fusion, test_transformer, bmm_reduction -- so this is that crash in one file that runs from run.py. Fixing the error path is what makes those four diagnosable at all.

LAYOUT PIN. `expect_ir` holds `torchsim.copy` in 08-passes/04-torchsim-materialize-layouts.mlir at 0: the layouts come down decided, so materialize-layouts writes no copy of its own.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sort_row_tl_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.sort(x, dim=1))

    return sort_row_tl_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sort(dim=1).values}


SPEC = KernelSpec(
    name="sort_row_tl_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-passes/04-torchsim-materialize-layouts.mlir": {"torchsim.copy": 0}},
)
