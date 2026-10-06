"""A gemm epilogue's bias row, over a lane axis deeper than one element.

    bias  memref<256xf32>       the row, one value per N
    tile  memref<32x256xf32>    lane_axis = 1, so 256 N over 128 lanes
    load  every ROW reads the same N addresses -- DRAM row stride 0

WHAT IT PINS, and it failed for exactly this reason before triton_shared
learned to structure a pointer broadcast:

    then     FAIL  max_abs_err 5.07693.  The load lowered to an INDIRECT
                   transfer, dram_stride [0, 0], with a 32x256 i64 index tile
    after    PASS  max_abs_err 7.62939e-06, dram_stride [0, 1], no gather

and the failure was not noise but a clean repetition: columns 0..127 got the
right bias and columns 128..255 got bias[0..127] AGAIN. Each lane fetched one
element for the row and wrote it into BOTH of the slots it owns, instead of
fetching a second index for the second slot. At BN = 128 a lane owns one slot
and the defect has nowhere to show, which is why that width passed throughout.

THE NUMBER NO LONGER SAYS IT, AND expect_ir DOES. Measured 2026-08-28: take
rewriteBroadcastOp's dispatch arm out today and this kernel still returns
7.62939e-06. The staging below learned to place a replicated tile in between, so
the gather now computes the RIGHT answer -- by fetching a 32x256 index tile to
read 256 distinct values. The defect the number caught has been fixed twice
over; the one this file is for has not moved, and only the IR still reports it:

    without the arm   tts.gather 1   tts.make_tptr 3   tts.load 2
    with it           tts.gather 0   tts.make_tptr 4   tts.load 3

That is what the expect_ir on the spec below holds, and it is the whole witness
now. A green number here means nothing about the broadcast arm.

WHY IT WAS A GATHER AT ALL. Inductor spells an addmm epilogue as a full
[BM, BN] load whose pointer is a BROADCAST pointer tensor:

    %p = tt.addptr(splat(bias_ptr), expand_dims(offs_n))   : 1x256
    %q = tt.broadcast %p                                   : 1x256 -> 32x256
    %r = tt.load %q

PtrAnalysis modelled that correctly -- it built `tts.make_tptr sizes [1, 256]
strides [0, 1]` -- but nothing rewrote the BROADCAST itself, so `ptrMap` had no
entry at [32, 256] and the load fell through to the unstructured path.
`rewriteBroadcastOp` supplies the missing arm. A pointer broadcast is a stride-0
dimension, which is true for any consumer of this IR, so it belongs there rather
than here (rule 11).

THE SPELLING IS LOAD-BEARING and getting it wrong cost three runs. A rank-1
`tl.load(bias_ptr + offs_n)` followed by `bias[None, :]` PASSED at BN = 512 even
before the fix -- it is a rank-1 descriptor and never became a gather. This
kernel uses the replicating form ON PURPOSE because that is the one Inductor
emits.

WHERE IT CAME FROM. PyTorchSim's Triton route capped the mm tiles it offered at
the lane count and tests/ops/attention/test_gqa.py was what the cap held up.
Running each suspect alone at BLOCK_N = 512, M = 32, K = 768, N = 1536:

    matmul(a, b)             6.96e-05      the wide tile alone is FINE
    matmul(a, bt.t())        5.53e-05      a transposed operand is FINE
    addmm(bias, a, b)        5.84          <-- the bias
    addmm(bias, a, bt.t())   5.84

With the fix in, gqa passes with the cap LIFTED, and the cap is gone.

WHY A MATMUL IS HERE. Without one the tile banks on the axis the row runs along
and the replication is an ordinary in-lane one, with nothing to stage. It is the
matmul demanding N on the lanes that makes a lane hold elements of a row it
needs all of -- two of them at BN = 256, in two different banks.

WHAT BREAKS IT AGAIN: anything that stops a broadcast pointer tensor reaching
`tts.make_tptr` at the broadcast's own shape, or any staging of that tile that
assumes the lane axis carries exactly one element.

A REPLICATING gemm bias row (dram row stride 0) over a lane axis carrying more than one element per lane. 128 passes, 256 and 512 fail. Isolated from test_gqa.py: of the wide tile, the transpose and the bias, only the bias moved -- 5.84 against 5.5e-05.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 32, 256, 128
BM, BN, BK = 32, 256, 128        # BN = 2 x the 128 lanes: the smallest failing width


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_bias_row_deeper_than_one_per_lane_kernel(
            a_ptr, b_ptr, bias_ptr, c_ptr,
            M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b)

        # THE EPILOGUE, SPELLED THE WAY INDUCTOR SPELLS IT. Not a rank-1 load
        # and a `[None, :]`, but a full [BM, BN] load whose index is the row
        # broadcast over the rows -- every row reads the same N addresses, so
        # the transfer's row stride is 0 and the load REPLICATES.
        #
        #   tmp0 = tl.load(in_ptr0 + tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N]))
        #
        # The rank-1 spelling passes at BN = 512 and a three-program grid; this
        # one does not, which is why the kernel is written this way round.
        acc = acc + tl.load(bias_ptr
                            + tl.broadcast_to(offs_n[None, :], (BM, BN)))

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return tile_bias_row_deeper_than_one_per_lane_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g, dtype=torch.float32),
            "b": torch.randn(K, N, generator=g, dtype=torch.float32),
            "bias": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"]) + inputs["bias"]}


SPEC = KernelSpec(
    name="tile_bias_row_deeper_than_one_per_lane_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "bias_ptr": "*fp32",
               "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("bias", IN, "float32", (N,)), Arg("c", OUT, "float32", (M, N))],
    grid=(1, 1),
    reference=reference, make_inputs=make_inputs,
    # Same reason as tests/cases/smoke/gemm.py: the accumulation order is not torch's.
    rtol=1e-3, atol=1e-3,
    # THE NUMBER STOPPED SAYING IT. This kernel returned 5.07693 without
    # rewriteBroadcastOp and it does not any more -- the staging below learned
    # to place a replicated tile, so the gather now computes the right answer
    # slowly. What the fix does is keep the access STRUCTURED, and only the IR
    # still says whether it did: 1 tts.gather without the arm, 0 with it.
    expect_ir={"03-ttshared.mlir": {"tts.gather": 0, "tts.make_tptr": 4,
                                    "tts.load": 3}},
)
