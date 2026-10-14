"""Flux.1's FluxSingleTransformerBlock -- ONE wide GEMM feeding attention and the
MLP side by side, concatenated back into one projection.

    wide      = x @ W1                       [T, 3D + 4D]   ONE matmul
    q, k, v   = wide[:, 0:D], [:, D:2D], [:, 2D:3D]
    mlp       = wide[:, 3D:3D+4D]
    attn      = softmax(q k^T / sqrt(D)) v                  [T, D]
    cat       = concat(attn, gelu(mlp))                     [T, 5D]
    y         = x + gate * (cat @ W2)

WHY THE BLOCK IS SHAPED LIKE THIS. Flux's 38 "single" blocks are the majority of
the model, and Black Forest Labs merged the qkv projection and the MLP's up
projection into ONE `linear1`, then merged the attention out-projection and the
MLP's down projection into ONE `linear2`. Nothing about the math changes -- two
block-diagonal GEMMs are one dense GEMM when the operands share `x` -- but the
block goes from four matmuls with four launches and four reads of the hidden
state to two. At Flux's 3072 width with mlp_ratio 4 the fused `linear1` is
[3072, 21504], the single largest weight in the model, which is why loading it
once matters more than the arithmetic does.

THE FUSION TOPOLOGY IS THE OPPOSITE OF swiglu.py's. There, TWO parallel matmuls
read one `x` tile and their results meet ELEMENTWISE -- a diamond that closes on
a multiply. Here ONE matmul produces a single wide tile and FOUR consumers take
disjoint COLUMN RANGES of it, three of which (q, k, v) feed a matmul while the
fourth feeds a pointwise nonlinearity. The pressure is on the SPLIT rather than
on the join: one producer, four readers at four offsets, with incompatible
downstream demands -- k has to arrive transposed into the matrix unit, mlp
has to stay exactly where it is.

  A Triton value cannot be sliced, so the wide tile is SPILLED and re-read at
  four column offsets. That is not a workaround, it is what the production kernel
  does with shared memory; the split is a re-address and never a data movement.
  It is also the only way to make the four readers visible to the backend as four
  `tts.make_tptr`s off one base with the same row stride (WIDE) and four scalar
  offsets -- conv2d.py's aliasing shape for nine shifted taps, except these tiles
  are disjoint rather than overlapping.

AND THE CONCAT IS REAL, not two row-blocked GEMMs. `concat(attn, gelu(mlp)) @ W2`
equals `attn @ W2[:D] + gelu(mlp) @ W2[D:]`, and writing it that way would erase
the thing worth testing: two tiles of DIFFERENT WIDTHS (D and 4D), produced by
two different unit types, written adjacent into one buffer that is then read back
as a single [T, 5D] operand. The store offsets are the only place the two widths
meet, and they are the whole join.

THE FREE AXIS IS NOT A POWER OF TWO, AND IT CANNOT BE. `tl.arange` demands one,
while 3D + 4D = 7D and D + 4D = 5D are never powers of two for any D -- so the
two full-width tiles are carried at the next power of two with a MASKED TAIL
(`rw < WIDE`, `other=0`), and the matmul's N axis runs over padding that
contributes zero. Every real Flux width hits this; a suite that only ever picked
round widths would never see the masked tail on a matmul operand at all. The four
column ranges themselves (D = 16 and 4D = 64) are exact, so only the producer and
the joiner pay for it.

  `k` is transposed on the way in -- `rt[None, :] * WIDE + (D + rd)[:, None]`,
  the strided read attention.py uses -- so no relayout pass is asked for.

GELU IS TANH-APPROXIMATE because that is what `nn.GELU(approximate="tanh")` in
the block is, and it is written out as `2 sigmoid(2u) - 1` rather than called:
that form saturates to -1 correctly when `exp(-2u)` overflows, where the textbook
`(e^2u - 1) / (e^2u + 1)` gives a NaN.

NOT MODELLED: the AdaLN-Zero modulation that produces `gate` (adaln_zero.py), the
per-head QK-RMSNorm (qk_norm.py) and the 2-D RoPE on q and k (rope_2d.py). Each
already has its own file; `gate` is taken as a given per-channel vector so the
residual is still gated and a fresh block is still the identity.

One wide GEMM with four disjoint column-range readers, three feeding a matmul and one a nonlinearity, joined by a REAL concat of two different widths read back as one operand -- over a free axis of 7D, which is never a power of two, so the producer and the joiner both carry a masked tail. `wide` is the GEMM alone, so a mismatch there is the matmul or the tail mask; a mismatch in `cat` is the split.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T = 16                  # tokens (image patches and text, one merged stream)
D = 16                  # model width == head dim, one head
MH = 4 * D              # 64 -- Flux's mlp_ratio is 4
WIDE = 3 * D + MH       # 112 -- linear1's output: qkv AND the mlp up
CAT = D + MH            # 80  -- linear2's input: attn AND the mlp act
WP = 128                # WIDE rounded up: arange needs a power of two
CP = 128                # CAT rounded up, likewise
SCALE = 0.25            # 1 / sqrt(D)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def flux_single_block_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, g_ptr,
                                 wide_ptr, cat_ptr, y_ptr,
                                 T: tl.constexpr, D: tl.constexpr,
                                 MH: tl.constexpr, WIDE: tl.constexpr,
                                 CAT: tl.constexpr, WP: tl.constexpr,
                                 CP: tl.constexpr, SCALE: tl.constexpr):
        rt = tl.arange(0, T)
        rd = tl.arange(0, D)
        rm = tl.arange(0, MH)
        rw = tl.arange(0, WP)                      # padded; tail masked off
        mw = rw < WIDE

        x = tl.load(x_ptr + rt[:, None] * D + rd[None, :])            # [T, D]

        # ONE WIDE GEMM. Everything below is a column range of this tile. The
        # last WP - WIDE columns of the operand are masked to zero, so the
        # matmul's free axis runs over padding that contributes nothing.
        w1 = tl.load(w1_ptr + rd[:, None] * WIDE + rw[None, :],
                     mask=mw[None, :], other=0.0)                     # [D, WP]
        wide = tl.dot(x, w1) + tl.load(b1_ptr + rw, mask=mw,
                                       other=0.0)[None, :]
        tl.store(wide_ptr + rt[:, None] * WIDE + rw[None, :], wide,
                 mask=mw[None, :])

        # FOUR READERS, FOUR COLUMN OFFSETS, ONE ROW STRIDE -- and these are
        # exact, no mask. `k` comes back transposed straight from DRAM so the
        # array gets the operand it wants without a relayout.
        q = tl.load(wide_ptr + rt[:, None] * WIDE + (0 * D + rd)[None, :])
        kt = tl.load(wide_ptr + rt[None, :] * WIDE + (1 * D + rd)[:, None])
        v = tl.load(wide_ptr + rt[:, None] * WIDE + (2 * D + rd)[None, :])
        m = tl.load(wide_ptr + rt[:, None] * WIDE + (3 * D + rm)[None, :])

        # BRANCH A: attention, [T, D].
        s = tl.dot(q, kt) * SCALE
        p = tl.exp(s - tl.max(s, axis=1)[:, None])
        attn = tl.dot(p / tl.sum(p, axis=1)[:, None], v)

        # BRANCH B: the MLP activation, [T, MH]. tanh-approximate GELU, written
        # as 2*sigmoid(2u) - 1 so a saturating exp gives -1 and not a NaN.
        u = 0.7978845608028654 * (m + 0.044715 * m * m * m)
        act = 0.5 * m * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0))

        # THE CONCAT, materialised: two tiles of DIFFERENT WIDTHS written
        # adjacent, then read back as ONE [T, CAT] operand.
        tl.store(cat_ptr + rt[:, None] * CAT + rd[None, :], attn)
        tl.store(cat_ptr + rt[:, None] * CAT + (D + rm)[None, :], act)

        rc = tl.arange(0, CP)
        mc = rc < CAT
        cat = tl.load(cat_ptr + rt[:, None] * CAT + rc[None, :],
                      mask=mc[None, :], other=0.0)                    # [T, CP]
        w2 = tl.load(w2_ptr + rc[:, None] * D + rd[None, :],
                     mask=mc[:, None], other=0.0)                     # [CP, D]

        out = tl.dot(cat, w2)
        tl.store(y_ptr + rt[:, None] * D + rd[None, :],
                 x + tl.load(g_ptr + rd)[None, :] * out)

    return flux_single_block_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32) * 0.5,
            "w1": torch.randn(D * WIDE, generator=g, dtype=torch.float32) * 0.2,
            "b1": torch.randn(WIDE, generator=g, dtype=torch.float32) * 0.1,
            "w2": torch.randn(CAT * D, generator=g, dtype=torch.float32) * 0.1,
            "g": torch.randn(D, generator=g, dtype=torch.float32) * 0.3}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    wide = x @ inputs["w1"].reshape(D, WIDE) + inputs["b1"][None, :]

    q = wide[:, 0 * D:1 * D]
    k = wide[:, 1 * D:2 * D]
    v = wide[:, 2 * D:3 * D]
    m = wide[:, 3 * D:3 * D + MH]

    attn = torch.softmax((q @ k.t()) * SCALE, dim=1) @ v
    act = torch.nn.functional.gelu(m, approximate="tanh")

    cat = torch.cat([attn, act], dim=1)
    y = x + inputs["g"][None, :] * (cat @ inputs["w2"].reshape(CAT, D))
    return {"wide": wide.reshape(-1).contiguous(),
            "cat": cat.reshape(-1).contiguous(),
            "y": y.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="flux_single_block_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w1_ptr": "*fp32", "b1_ptr": "*fp32",
               "w2_ptr": "*fp32", "g_ptr": "*fp32", "wide_ptr": "*fp32",
               "cat_ptr": "*fp32", "y_ptr": "*fp32", "T": "constexpr",
               "D": "constexpr", "MH": "constexpr", "WIDE": "constexpr",
               "CAT": "constexpr", "WP": "constexpr", "CP": "constexpr",
               "SCALE": "constexpr"},
    constexprs={"T": T, "D": D, "MH": MH, "WIDE": WIDE, "CAT": CAT,
                "WP": WP, "CP": CP, "SCALE": SCALE},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("w1", IN, "float32", (D * WIDE,)),
          Arg("b1", IN, "float32", (WIDE,)),
          Arg("w2", IN, "float32", (CAT * D,)),
          Arg("g", IN, "float32", (D,)),
          Arg("wide", OUT, "float32", (T * WIDE,)),
          Arg("cat", OUT, "float32", (T * CAT,)),
          Arg("y", OUT, "float32", (T * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
