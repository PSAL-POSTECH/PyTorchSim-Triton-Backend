"""A conv epilogue's row tail stores OUTSIDE the tensor, and the wrapper hides it.

WHAT THIS PINS. PyTorchSim's conv template flattens (n, h, w) into one row axis
and decomposes it back with `%` and `//`. When the row block does not divide the
feature map, the last program stages rows that do not exist, and the store for
those rows leaves the buffer:

    User store segfault @ 0xd1158130

The guard that should stop them is `idx_n < BATCH` -- a bound on the OUTERMOST
quotient, `nhw // OUT_W // OUT_H`, not on `nhw`. The other two row terms
(`idx_y_h < OUT_H`, `idx_y_w < OUT_W`) are true by construction. The transfer
that comes out clamps the COLUMN axis and nothing else:

    "torchsim.transfer"(...) {dma_kind = "MVOUT", dram_stride = [0, 784],
        indirect, masked_axes = [1], masked_fill = 0}

and the column axis is full at 128 of 128, so that clamp can never fire.

HOW IT WAS NARROWED, from resnet18's 128->128 conv over 28x28 (784 rows,
BLOCK_M 64, grid 13, so 832 rows are staged and 48 do not exist). Each step
below still faulted at the same address, so each thing removed is NOT part of
the cause:

    the whole model                     segfault
    the kernel alone, from the dump     segfault @ 0xd1161430
    tl.dot removed, both loads kept     segfault @ 0xd1161430
    the entire K loop removed           segfault @ 0xd1158130

and one step turns it off, which is what names the cause:

    BLOCK_M 64 -> 16 (49 blocks, 784 = 49*16, no tail)   ALL STAGES PASSED

So it is the tail and only the tail. Not the matmul, not the loads, not the
loop.

THE TWO MASK TERMS THAT LOOK REDUNDANT ARE NOT. `idx_y_h < OUT_H` and
`idx_y_w < OUT_W` always hold, and a hand-written kernel that drops them --
keeping the decomposition, the column clamp, the grid and the exact 784x128
shape -- PASSES. They are carried here verbatim for that reason: the shape that
reproduces is the shape the template emits, redundant comparisons included.

WHAT WAS RULED OUT. The descriptors are identical between the faulting and the
passing BLOCK_M, so it is not the descriptor. Every indirect index value is in
range in all 13 programs (max 100351 of 100352), so it is not a bad index.

WHY IT HAS NEVER BEEN SEEN. `pytorchsim_triton_compiler/lowering/wrapper.py` allocates every argument at TWICE
its size, and the store lands in the second half. The factor reads as a tail
guard -- the comment says so, and the original spelling credited PyTorchSim,
whose `__wrap_malloc` rounds up to 512 bytes and does not double. Measured:
test_resnet.py and test_vit.py pass at `* 2`, fail at 512-byte alignment and at
a 64 KiB fixed pad, and pass again at 256 KiB. A size threshold, not a tail.
That factor is also what stops Llama 4 Scout, whose 1.25 GiB expert stack is
requested as 2.5 GiB and fails to allocate; with it removed Scout runs at its
real 5120 width to 7.0482e-06.

WHAT WOULD BREAK THIS. Reading `n < BATCH` as a bound on the row axis and then
dropping it again; attaching the row clamp to the column axis (rule 18 -- a
mask names an AXIS and the name has to travel); or folding the redundant
comparisons in a way that takes the real one with them.

THE FIX LANDED, and the sentence above is what it does: a bound on `row // K`
IS a bound on `row`, so MaskAnalysis reads `(x // K) < c` as `x < c*K` rather
than refusing the quotient. Three identities go with it -- parseCmp folds
`(x % K) < K` away as a term that cannot be false, c01's _drop_masked_modulo
drops a modulo the access's own mask already bounds away, and its
reflattenDivRem (PtrAnalysis.cpp) puts `(x % K) + K*(x // K)` back together. The store then comes out
structured: `tts.make_tptr`, `dram_stride [1, 784]`, `masked_axes [0, 1]`, so
the row guard reaches the transfer instead of the column axis alone.

resnet18's conv epilogue, stripped to the store. Its tail rows used to leave the buffer; the four ttclean identities put the guard on the row axis, so the transfer clamps [0, 1] instead of [1].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, OUT

OUT_H = OUT_W = 28       # resnet18's 28x28 feature map -> 784 rows
OUT_C = 128
BATCH = 1
BLOCK_M = 64             # 784 = 12*64 + 16, so the 13th block stages 48 dead rows
BLOCK_N = OUT_C
PROGRAMS = (OUT_H * OUT_W + BLOCK_M - 1) // BLOCK_M
NUMEL = OUT_C * OUT_H * OUT_W


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def conv_tail_store(out_ptr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
                        BATCH: tl.constexpr, OUT_C: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        INDEX_DTYPE: tl.constexpr = tl.int32
        # The template's own decomposition, verbatim.
        nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
        idx_y_w = nhw % OUT_W
        nh = nhw // OUT_W
        idx_y_h = nh % OUT_H
        idx_n = nh // OUT_H
        idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # FOUR terms. The middle two always hold -- and dropping them makes this
        # kernel pass, so they stay.
        mask = (
            (idx_n < BATCH)[:, None]
            & (idx_y_h < OUT_H)[:, None]
            & (idx_y_w < OUT_W)[:, None]
            & (idx_y_c < OUT_C)[None, :]
        )
        idx_c = idx_y_c[None, :]
        idx_h = idx_y_h[:, None]
        idx_w = idx_y_w[:, None]
        tl.store(out_ptr + tl.broadcast_to(
            idx_w + OUT_W * idx_h + (OUT_H * OUT_W) * idx_c,
            [BLOCK_M, BLOCK_N]), acc, mask)

    return conv_tail_store


def make_inputs(torch, seed=0):
    return {}


def reference(inputs):
    import torch
    return {"o": torch.zeros(NUMEL, dtype=torch.float32)}


SPEC = KernelSpec(
    name="conv_tail_store", kernel=kernel,
    signature={"out_ptr": "*fp32", "OUT_H": "constexpr", "OUT_W": "constexpr",
               "BATCH": "constexpr", "OUT_C": "constexpr",
               "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
    constexprs={"OUT_H": OUT_H, "OUT_W": OUT_W, "BATCH": BATCH,
                "OUT_C": OUT_C, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
    args=[Arg("o", OUT, "float32", (NUMEL,))],
    grid=(PROGRAMS, 1, 1), reference=reference, make_inputs=make_inputs,
)
