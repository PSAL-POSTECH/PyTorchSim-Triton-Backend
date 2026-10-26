"""Indirect transfers whose index tensors bufferize into ONE allocation.

Both operands of the matmul are wrapped -- `offs % M` on the rows of A and
`offs % N` on the columns of B -- and the STORE wraps both axes at once, so
convert_accesses_to_transfer lowers all THREE as indirect transfers, each with its own
index tile. The tiles have the same type and disjoint live ranges (fill A's
indices, gather A, fill B's indices, gather B, ...), so one-shot-bufferize gives
them ONE `memref<128x128xi64>`. That is bufferize doing its job; the damage is
what it does to the axis graph.

THE STORE JOINED ON 2026-08-31, when `shape` started carrying one wrap per axis
instead of one in total; before that it was a tts.scatter and stayed out of the
axis graph. So the collapse below started firing TWICE -- two round trips, both
POINTLESS at the time -- on the same unchanged verdict. The answer did not move
(9.53674e-06 either way); the surface of the defect did. It costs nothing today:
`expect_ir` below pins `pytorchsim_tpu.xlu_move` at 0 in 08-transformed.mlir --
the bus this once cost a trip on is gone, and the unit that replaced it is what
a layout is paid for with now.

WHY IT COLLAPSES. lib_axis_graph relates an indirect transfer's index operand to
its tile axis-for-axis, and it must: the DMA reads the index at the same banked
slot it writes the gathered element to. With ONE index buffer that edge runs
through BOTH transfers, so A's axes are related to B's THROUGH it -- and the
matmul calls A(m, k) and B(k, n), so m ~ k ~ n and the whole iteration space
lands in one class. A class holding two axes of the same buffer has no axis to
give, and `axis_of` falls back to the lower one:

    shared      index buffer lane_axis 0, both f32 tiles lane_axis 1
    split       index buffers lane_axis 1, both f32 tiles lane_axis 1

WHAT IT COSTS, and it is not a refusal. Everything below the stamp is correct --
the descriptors, the `dram_stride` of [0, 1] and [128, 0], and the index values
themselves, which read exactly `r * 128` and `c` out of the runtime dump. A
0-banked index feeding a 1-banked tile gathers the right addresses into the
wrong lanes, so the kernel compiles clean, runs to completion, and answers
55.0492 away from torch on a 128x128x128 f32 mm.

WHAT FIXES IT is fit_layout_to_lanes's `_split_generations`, which mints one allocation per live
range. Its keep/undo verdict used to compare the two ranges to EACH OTHER, see
both wanted axis 1, call that no gain and put the collapse back; the question a
split has to be asked is whether it changes an answer at all, and 1 against the
shared node's 0 is a change. That half is fixed and the numbers have been right
since.

THE SPLIT THEN STOPPED BEING OFFERED, and the answer stayed right for a worse
reason. Bufferize now writes each generation of the shared buffer through a COPY
into an index buffer of its own, so the shared allocation is not ITSELF any
transfer's index and `_find_shared_index_allocs` -- which asks exactly that --
returned nothing. The copy relates the axes just the same, so the collapse
survived one hop further out, and `_cut_gather_index` repaired it instead, at two
DRAM round trips, both POINTLESS then, both ends lane axis 1. The repair works
because a `pytorchsim.relayout` is in NO_EDGE, which separates the classes rather than
moving anything on the machine.

So the allocation is now OFFERED to the split when its axis CLASS holds two
indirect transfers' indexes, and kept because the verdict says the answer moved.
Offered and not required: with a copy in the way each index is its own
allocation and can carry its own layout, so the collapse costs a round trip
rather than an answer, and the verdict is the right judge.

WHAT BREAKS IT AGAIN: any verdict that only asks whether two live ranges of one
allocation disagree with each other; any reader that relates an indirect
transfer's index to its tile without noticing that one buffer can be two tiles;
and any test for "is this the shared index" that asks whether a value IS one
rather than whether it is related to one.

SHAPES ARE LOAD-BEARING. M = N = K = BM = BN = BK = 128 is what makes the two
index tensors the same type, single-trip, and adjacent -- which is what makes
bufferize share the allocation at all. Changing any of them can stop reproducing
without the defect being fixed.

Both mm operands wrapped, so both load through an indirect transfer and bufferize shares ONE index allocation between them. Answers 55.0492 without the generation split -- the index buffer is stamped lane_axis 0 while both tiles it feeds are 1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 128, 128, 128
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_two_gathers_one_index_buffer_kernel(
            a_ptr, b_ptr, c_ptr,
            M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        # The wrap is what makes each load indirect. It is also what Inductor's
        # mm template emits (`rm % M` / `rn % N`), so this is the shape the
        # Triton route actually hands the backend, not a contrivance.
        offs_m = (pid_m * BM + tl.arange(0, BM)) % M
        offs_n = (pid_n * BN + tl.arange(0, BN)) % N
        offs_k = tl.arange(0, BK)

        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return ptr_two_gathers_one_index_buffer_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g, dtype=torch.float32),
            "b": torch.randn(K, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="ptr_two_gathers_one_index_buffer_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(M // BM, N // BN),
    reference=reference, make_inputs=make_inputs,
    # Same reason as tests/cases/smoke/gemm.py: the accumulation order is not torch's.
    rtol=1e-3, atol=1e-3,
    # THE NUMBER CANNOT SAY IT. Both repairs answer 9.53674e-06; only the count
    # tells the free one from the one that crosses DRAM twice to change nothing.
    expect_ir={"08-transformed.mlir": {"pytorchsim_tpu.xlu_move": 0, "pytorchsim.distributed_reduction": 0}},
)
