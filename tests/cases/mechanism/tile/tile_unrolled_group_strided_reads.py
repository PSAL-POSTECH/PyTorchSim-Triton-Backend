"""One buffer read at FOUR offsets of one stride, plus a `//` and a `%` beside it.

Inductor does not always emit a reduction for a reduction. A mean over four
adjacent elements is short enough that it unrolls the fold into four strided
loads and three adds, and what arrives here has no reduction axis at all:

    tmp0 = tl.load(in_ptr0 + (4*x))
    tmp1 = tl.load(in_ptr0 + (1 + 4*x))
    tmp3 = tl.load(in_ptr0 + (2 + 4*x))
    tmp5 = tl.load(in_ptr0 + (3 + 4*x))
    mean = (tmp0 + tmp1 + tmp3 + tmp5) / 4.0

FOUR VIEWS OF ONE BUFFER, same stride, four consecutive start offsets, live at
once and added together. Every other multi-read kernel in this suite reads
DIFFERENT buffers, or one buffer through one view; nothing pins what happens
when one buffer is staged four ways in a single elementwise op.

    measured, on the tree that first ran it: all four come out as INDIRECT
    transfers -- four gathers, `ind=4` on 04-transformed.mlir -- although
    `4*x + k` is affine in x and a strided descriptor with a start offset says
    the same thing for free. That is not wrong and this kernel does not assert
    it is; it is recorded because a gather reads an index buffer per element
    where a box reads none, and because the class it lands in is the one that
    faulted next door (mask_tail_row_address_out_of_range).

IT FAILS, AND NOT FOR THE REASON THE FILE WAS WRITTEN FOR. The tile is XBLOCK
256 on a 128-lane machine, so each lane holds TWO elements, and every element in
the second slot comes back wrong -- 128..191 of 192 here, and 128..255 of 256
when COLS is widened to 128 so there is no masked tail at all. So it is the
DEPTH, not the tail: the first per-lane slot is right and the second is garbage
that matches no index shift of any operand.

    XBLOCK 256, N 192   bad = 128..191    (64 of 192, all in slot 2)
    XBLOCK 256, N 256   bad = 128..255    (128 of 256, all of slot 2)

Every block size in kernels/coverage is 32, 64 or 128, so nothing here had ever
put more than one element in a lane -- which is why the suite could not see
this. `tile_deeper_than_one_per_lane.py` and `tile_gemm_lane_axis_deeper.py`
(added on develop-select-grid for exactly that gap) both PASS on this tree, so
depth alone is not it either; what this file adds on top of them is the four
strided views of ONE buffer.

IT IS LEFT FAILING, WITH NO `expect` MARKER, AND THAT IS THE HONEST LABEL. The
three markers mean: pass, a refusal this compiler DECIDES, or a block that is
"a fact about somebody else's tree and WILL lift one day". This is none of them
-- every stage produces a program and our own lowering returns wrong numbers --
and the runner says so out loud if you claim otherwise ("the ANSWER is what
failed, which is a wrong number, not a refusal"). So it sits in the sweep as a
failure, which is what an undiagnosed defect of ours is.

NOT THE GATHER CLAMP, verified rather than assumed: run on 14030f8, before that
fix landed, it fails with the identical 4.81383 and the same 64 elements.

AND TWO MORE ADDRESSINGS IN THE SAME KERNEL. The head this came from also
decomposes the flat index for two other operands:

    x1 = x // 768      a per-row statistic, broadcast along the row
    x0 = x % 768       a per-column weight, broadcast down the rows

so one elementwise op reads five operands through four different index
expressions. The `//` and the `%` are the pair that, on a masked tail, walked
off the end in convnextv2's kernel 40. Here they are in range for every real
row, so what is under test is the OPPOSITE case: the same arithmetic where the
addresses are fine and the answer still has to be right.

WHERE IT CAME FROM. convnextv2's last kernel,
`triton_npu_fused_mean_native_layer_norm_50` -- the classifier head, where a
global mean over the 2x2 spatial tail meets the final LayerNorm.

WHAT WOULD BREAK THIS
  * folding the four views into one and reading it once with the wrong start
    offset: the four differ ONLY in that offset, so an off-by-one is invisible
    in three of them and the sum is still four numbers.
  * broadcasting `x // 768` down the wrong axis, which swaps the row statistic
    with the column weight. Both are [C]-shaped here on purpose -- ROWS and
    COLS are different so a transposed read cannot pass by luck.

convnextv2's head: a four-element mean unrolled into four strided reads of one buffer, beside a `//` row statistic and a `%` column weight -- five operands through four index expressions in one op.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS = 2             # what `x // COLS` selects
COLS = 96            # what `x % COLS` selects
N = ROWS * COLS      # 192 against XBLOCK 256: a 64-row tail, and depth 2
GROUP = 4            # the unrolled fold's width


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def unrolled_group_strided_reads(x_ptr, m_ptr, v_ptr, w_ptr, b_ptr, o_ptr,
                                     N: tl.constexpr, COLS: tl.constexpr,
                                     GROUP: tl.constexpr,
                                     XBLOCK: tl.constexpr):
        x = tl.arange(0, XBLOCK)
        xmask = x < N
        x1 = x // COLS
        x0 = x % COLS
        # THE UNROLLED FOLD: one buffer, one stride, four start offsets.
        a0 = tl.load(x_ptr + (GROUP * x), xmask)
        a1 = tl.load(x_ptr + (1 + GROUP * x), xmask)
        a2 = tl.load(x_ptr + (2 + GROUP * x), xmask)
        a3 = tl.load(x_ptr + (3 + GROUP * x), xmask)
        mean = (a0 + a1 + a2 + a3) / 4.0
        # A row statistic and a column weight, from the same flat index.
        m = tl.load(m_ptr + x1, xmask)
        v = tl.load(v_ptr + x1, xmask)
        w = tl.load(w_ptr + x0, xmask)
        b = tl.load(b_ptr + x0, xmask)
        y = (mean - m) * tl.rsqrt(v + 1e-12) * w + b
        tl.store(o_ptr + x, y, xmask)

    return unrolled_group_strided_reads


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(GROUP * N, generator=g, dtype=torch.float32),
            "m": torch.randn(ROWS, generator=g, dtype=torch.float32),
            "v": torch.rand(ROWS, generator=g, dtype=torch.float32) + 0.5,
            "w": torch.randn(COLS, generator=g, dtype=torch.float32),
            "b": torch.randn(COLS, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(N, GROUP)
    mean = x.mean(dim=1)
    idx = torch.arange(N)
    m = inputs["m"][idx // COLS]
    v = inputs["v"][idx // COLS]
    w = inputs["w"][idx % COLS]
    b = inputs["b"][idx % COLS]
    return {"o": (mean - m) * torch.rsqrt(v + 1e-12) * w + b}


SPEC = KernelSpec(
    name="unrolled_group_strided_reads", kernel=kernel,
    signature={"x_ptr": "*fp32", "m_ptr": "*fp32", "v_ptr": "*fp32",
               "w_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr", "COLS": "constexpr", "GROUP": "constexpr",
               "XBLOCK": "constexpr"},
    constexprs={"N": N, "COLS": COLS, "GROUP": GROUP, "XBLOCK": 256},
    args=[Arg("x", IN, "float32", (GROUP * N,)),
          Arg("m", IN, "float32", (ROWS,)), Arg("v", IN, "float32", (ROWS,)),
          Arg("w", IN, "float32", (COLS,)), Arg("b", IN, "float32", (COLS,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
