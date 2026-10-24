"""transpose and the no-reassociation reshape, pinned NUMERICALLY.

Both lower to a DRAM round trip (`pytorchsim.relayout`) and both were shipped without
a kernel that could reach them -- ops_shape, the file they were written for,
stopped two stages earlier on something else. One of them was wrong, and this
is the file that would have said so.

WHAT SEPARATES THE TWO

  A RESHAPE preserves row-major linear order: `[8, 2]` and `[2, 8]` are the same
  sixteen floats in the same order, so both ends write and read the tile
  LINEARLY and only the descriptors differ.

  A TRANSPOSE does not. Output (i, j) is input (j, i), so one image cannot be
  read two ways -- it is WRITTEN two ways instead. The MVOUT scatters the input
  into the output's image (input axis a takes the stride of the output axis the
  permutation sent it to) and the MVIN stays an ordinary read.

  The bug was in exactly that difference. The MVOUT's strides were sized by the
  OUTPUT for both, which is right only while the two shapes agree -- true for a
  plain rebanking relayout, false for a reshape. On `[8,2] -> [2,8]` it wrote
  element (i, j) at i*8 + j instead of i*2 + j and the tile came back scrambled,
  while the rebank and transpose cases stayed correct and hid it.

WHAT WOULD BREAK IT. Drop the permutation arm of `_shuffles_the_tile`
(fit_layout_to_lanes) and the `linalg.transpose` bufferize leaves behind rides all
the way to vectorize, which refuses it: "'linalg.transpose' still touches a
scratchpad buffer after banking". That is how this kernel failed before that arm
existed -- the question the pass asked was phrased as `o.results[0]`, and a
linalg op with buffer semantics has none.

IT CARRIES NO ROUND TRIP AT ALL NOW, AND THAT IS A COVERAGE HOLE (rule 2)
It used to carry two, and the transpose's was the avoidable one: BACKWARD it is
free, because the transpose's source has exactly one use and its producer is the
MVIN that staged the tile, so the whole permutation is one `dram_stride` -- [8,
1] over 2x8 becoming [1, 8] over 8x2. The DMA has been transposing on every load
all along, and absorb_layout_ops now takes it there: 08-transformed.mlir holds 1
MVIN and 2 MVOUT and no `pytorchsim.relayout` at all.

So this kernel no longer reaches `_shuffles_the_tile`'s permutation arm, which
is where the MVOUT-stride bug above lived. It passes without exercising what it
was written to pin; ops_turned_copy_is_a_shuffle.py holds that arm now, and it
gets there the only way left -- a copy the pass stages and then turns, since no
`linalg.transpose` survives absorb_layout_ops anywhere in the suite. What this
file still pins is the numbers.
torch equivalent: x.t(), then that viewed as (2, 8).

The MVOUT of a relayout takes its strides from the INPUT's shape unless a permutation says otherwise; sizing them by the output is correct only when the two shapes agree.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K = 2, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_relayout_shapes_kernel(x_ptr, t_ptr, r_ptr,
                                   M: tl.constexpr, K: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])       # (2, 8)
        t = tl.trans(x, 1, 0)                                     # (8, 2)
        tl.store(t_ptr + rk[:, None] * M + rm[None, :], t)
        d = tl.reshape(t, (M, K))            # (8,2) -> (2,8), no reassociation
        tl.store(r_ptr + rm[:, None] * K + rk[None, :], d)

    return ops_relayout_shapes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, K)
    t = x.t().contiguous()
    return {"t": t.reshape(-1), "r": t.reshape(M, K).reshape(-1)}


SPEC = KernelSpec(
    name="ops_relayout_shapes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "t_ptr": "*fp32", "r_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "K": K},
    args=[Arg("x", IN, "float32", (M * K,)), Arg("t", OUT, "float32", (M * K,)),
          Arg("r", OUT, "float32", (M * K,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
