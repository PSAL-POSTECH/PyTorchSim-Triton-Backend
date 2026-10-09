"""A gather whose index is a remainder reads the address without the remainder.

    row  = (m * 7) % M                  an index along axis 0 only
    keep = row < M - 3                  a mask along axis 0 only
    o[m, n] = keep[m] ? x[row[m] + n * M] : 0

THE MODULO IS DROPPED FROM THE ADDRESS AND KEPT IN THE MASK. `07-regularized.mlir`
says it outright:

    %12 = linalg.index 0                          m, 0..63, raw
    %13 = %12 * 7
    %14 = %13 % 64                                the row this kernel means
    %16 = (%14 < 61)                              the mask reads %14 -- correct
    %17 = tts.make_gather_scatter_tptr(%7, %12, %2, %1, %16)
                                           ^^^ the OFFSETS are %12, not %14

The linalg lowering hoisted the `* 7` out of the `arith.remsi` into the descriptor
stride and discarded the remainder, so convert_accesses_to_transfer receives
`offset_stride = 7` over the raw induction variable. The address computed is

    base + m*7        + n*64      instead of      base + (m*7 % 64) + n*64

MEASURED, AND THE ARITHMETIC CLOSES. m = 11 gives 11*7 = 77 where the row is
77 % 64 = 13, and 77 = 13 + 64 -- exactly one column on:

    o[11, 44] wants x[13 + 44*64] =  1.6937122
    o[11, 44] gets  x[13 + 45*64] = -4.093737    = x[77 + 44*64]
    3328 / 4096 elements outside tolerance

Every stage passes and the kernel runs. The mask being right while the address
is wrong is what makes it silent.

THE SAME DEFECT STOPS RESNET, at a different magnitude. Inductor's conv template
builds its spatial index with `nhw % OUT_W` and `nh % OUT_H` where nhw reaches
12544, so dropping those remainders gives indices in the millions -- measured as
a scratchpad address 9.7 MB above a buffer region 8.3 KB long, which spike
reports as

    User load segfault @ 0x00000000d0f66270

This kernel is that bug at a size small enough to read.

THAT IS ALL HISTORY, and this docstring described it in the present tense for
two repairs longer than it was true. What the file holds today:

  8085050   refused to rebuild a wrapped pointer as a gather -- make_gather_
            scatter_tptr has no `shape` field to keep the modulus in. Exact
            from then on.
  3115d94   sent a READ's unstructured mask to an arith.select instead of
            rebuilding the pointer at all, and this mask is on a read. So the
            refusal stopped being asked: measured over 396 kernels it fired 0
            times, and removing it moved no kernel and no IR byte.
  b0d2551   the select became the pass's own default and the guard went with
            it, unreachable.
  today     the mask rides tts.load's `elem_mask` instead of being turned into
            a select upstream, so 07-regularized holds the descriptor AND the mask
            and _empty_read_mask_carriers is what chooses. expect_ir below pins both.
  3.8       Triton 3.8 prices a register-only convert at 0, so remove-layout-
            conversions keeps the identity ttg.convert_layout on this mask (its
            remsi is shared with the address). MaskAnalysis then has to look
            through a convert as PtrAnalysis does, or the load gathers. This
            kernel is that witness: tts.gather 1 without the arm, 0 with it.

IT IS STILL WRONG UNDER `unstructured-mask-as-select=false` -- an option that
no longer exists; the demoting arm it switched on is gone, and this paragraph
is what the measurement said while it was there. Measured at
5.78745 with the guard removed. That path is what the four unstructured_mask_*
lit tests upstream pin, and this kernel no longer reaches it -- so it is not
that path's witness, and says so rather than looking like one.

WHY THE SHAPE IS WHAT IT IS. The other axis needs a NON-unit stride or the whole
access folds into one contiguous DMA and no gather descriptor is built at all.
The mask is what makes the two spellings of the row visible side by side; without
it the kernel is still wrong, but nothing in the IR shows the correct index being
computed and then ignored.

A masked read through a WRAPPED pointer. It must take the select path and leave the make_tptr alone -- rebuilding it as a gather drops the modulus from the address and reads one column over.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 64, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def masked_gather_column_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                    N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        row = (rm * 7) % M                       # computed, axis 0 only
        keep = row < (M - 3)                     # a mask along axis 0 only
        # rn's stride is M, NOT 1: a unit inner stride folds into the DMA's
        # contiguous dimension and there is no second axis left to get wrong.
        v = tl.load(x_ptr + row[:, None] + rn[None, :] * M,
                    mask=keep[:, None], other=0.0)
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], v)

    return masked_gather_column_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    rows = (torch.arange(M) * 7) % M
    cols = torch.arange(N) * M
    o = x[rows[:, None] + cols[None, :]]
    return {"o": (o * (rows < (M - 3))[:, None]).reshape(-1)}


SPEC = KernelSpec(
    name="masked_gather_column_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0.0, atol=0.0,
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 0,
                                    "tts.gather": 0},
               "08-passes/03-torchsim-convert-accesses-to-transfer.mlir":
                   {"arith.select": 1}},
)
