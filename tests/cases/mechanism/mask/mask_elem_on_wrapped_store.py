"""THE LAST EMPTY CELL: an elementwise mask on a WRAPPED structured store.

The address ops make a ladder, and the rung is how many axes are non-contiguous:
`tts.make_tptr` none, `tts.make_gather_scatter_tptr` exactly one, `tts.gather`
all of them. A mask does not change that number, so it must not move a pointer
down the ladder -- and since `structured-elem-mask` it does not: the mask rides
in `tts.store`'s own `elem_mask` and the descriptor keeps everything it had,
`shape` (the wraparound) included. This is the only kernel that spells the two
together ON A STORE, which is the only end that reaches convert_accesses_to_transfer -- and the store is
what the backend refused. Measured on the suite's 392 stage-3 dumps: a
`make_tptr` carrying an `elem_mask` appears in 4 kernels flat and 2 wrapped, and
of the wrapped pair the other one (tile_masked_gather_column) is a LOAD, whose
mask _empty_read_mask_carriers removes a stage before convert_accesses_to_transfer ever sees the descriptor. So the
cell looked occupied and the refusal was still unreachable.

WHY THE PAIR IS THE INTERESTING ONE. They are the two facts that have to live in
the SAME per-position index, and each wants the index for itself:

    the wrap    dim 0's contribution is `(off + p*stride) mod W`, so that axis's
                offset goes INSIDE the modulo and out of the transfer's flat
                offset (convert_accesses_to_transfer `_emit_wrap`)
    the mask    a masked-off position has no per-element write-enable, so it is
                aimed at the buffer's padding instead (convert_accesses_to_transfer `_park_masked_offsets`,
                wrapper doubles an `out`)

Before this kernel convert_accesses_to_transfer's `_emit_masked_structured` REFUSED any wrap outright --
"the index below would have to carry the modulo as well, which nothing pins yet"
-- and the refusal was honest exactly because nothing pinned it. Now the modulo
is computed first and the park is applied to the result, which is the only order
that works: park-then-wrap would fold the parking offset back into the buffer.

WHAT BREAKS IT.

    dropping the modulo         MEASURED: `verify: numerical mismatch 2.56723`.
                                Rows 16..47 are inside `o`, so the writes land
                                and no bound is violated -- 20 of the 48 rows
                                are simply wrong (32 rows that must stay zero,
                                five eighths of them flagged). This is why the
                                wrap extent (16) is SMALLER than the buffer
                                (48 rows): a wrap at the buffer's own size would
                                send a dropped modulo into the PADDING, where
                                the doubled `out` swallows it and the kernel
                                passes while the compiler is broken.
    parking before the wrap     `park % 16` is a live row; masked-off rows would
                                overwrite live ones.
    leaving offsets[0] in the   dim 0's offset would be counted twice, once
    transfer's flat offset      inside the modulo and once outside.
    a per-COLUMN flag instead   would be right too, but it makes the mask a
                                broadcast ACROSS the lane axis and the kernel
                                would then be red on vectorize for a reason
                                that has nothing to do with wrap
                                (mask_col_flag_broadcast_over_rows is that file).

WHAT IT DOES NOT PIN. `_emit_masked_structured` still refuses TWO wrapped dims,
which upstream cannot build either (PtrAnalysis rank check, pinned by
ptr_modulo_rank_reject.py). And the mask is only on the store: a load's
elem_mask never reaches convert_accesses_to_transfer, _empty_read_mask_carriers turns it into a select first
(mask_gs_from_structured.py pins that end).

torch equivalent: rows of `a` written into `o` at `row % 16`, but only the rows
whose runtime flag is set; everything else stays zero.

The one kernel where a wraparound and an elementwise mask share one descriptor. Both have to end up in the same per-position index -- the modulo first, the parking of masked-off positions after -- and before this the masked path refused any wrap outright.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

W = 16          # the wraparound extent, in ROWS -- smaller than the buffer
M = 48          # rows allocated in `a` and `o`
BM = 8          # rows per program
BN = 16
GRID = 6        # 6 * 8 = 48 rows walked, folded onto 16 by the wrap


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def elem_mask_on_wrapped_store(a_ptr, rf_ptr, o_ptr, W: tl.constexpr,
                                   BM: tl.constexpr, BN: tl.constexpr):
        pid = tl.program_id(0)
        p = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        # THE WRAP. `shape[0]` on the descriptor, not an arithmetic op.
        rm = (pid * BM + p) % W
        # THE MASK. Loaded from memory, so MaskState::parse cannot fold it into
        # `mask_dims`; it stays an i1 tensor and rides in `elem_mask`. Per ROW,
        # so it broadcasts ALONG the lane axis rather than across it.
        rowf = tl.load(rf_ptr + p) != 0
        v = tl.load(a_ptr + rm[:, None] * BN + rn[None, :])
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], v, mask=rowf[:, None])

    return elem_mask_on_wrapped_store


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M * BN, generator=g, dtype=torch.float32),
            "rf": (torch.arange(BM, dtype=torch.int32) % 3).contiguous()}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(M, BN)
    rf = inputs["rf"] != 0
    o = torch.zeros(M, BN, dtype=torch.float32)
    # The kernel's own loop. Duplicate rows are written with the same value --
    # `v` is read through the SAME wrapped pointer -- so the order does not
    # matter and this stays deterministic.
    for pid in range(GRID):
        for p in range(BM):
            if rf[p]:
                r = (pid * BM + p) % W
                o[r, :] = a[r, :]
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="elem_mask_on_wrapped_store", kernel=kernel,
    signature={"a_ptr": "*fp32", "rf_ptr": "*i32", "o_ptr": "*fp32",
               "W": "constexpr", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"W": W, "BM": BM, "BN": BN},
    args=[Arg("a", IN, "float32", (M * BN,)),
          Arg("rf", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (M * BN,))],
    grid=(GRID,),
    reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"03-linalg.mlir": {"tts.make_gather_scatter_tptr": 0,
                                    "tts.scatter": 0,
                                    "tts.make_tptr": 3}},
)
