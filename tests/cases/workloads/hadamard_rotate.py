"""QuaRot / SpinQuant `rotate_activation` -- a Hadamard transform over the feature
dim, done as a log-depth butterfly.

    for h in 1, 2, 4, ... D/2:                 log2(D) STAGES
        for each pair (i, i^h):
            a, b  <-  a + b, a - b
    out = x / sqrt(D)

WHY A ROTATION IS A QUANTISATION KERNEL
  Everything that makes activations hard to quantise is the outlier channels: a
  few feature dimensions run one to two orders above the rest, in every token,
  and one scale over the vector spends its whole range on them. SmoothQuant
  migrates the problem into the weights; QuaRot removes it instead. Multiply the
  activation by an orthogonal matrix Q and the weight by Q^T -- the layer
  computes the same thing, because (xQ)(Q^T W) = xW -- but xQ is a MIXTURE of all
  D channels, so an outlier in one channel is spread over all of them and the
  vector becomes roughly Gaussian with no spikes at all. QuaRot reports 4-bit
  weights AND 4-bit activations from this, which per-channel scaling alone does
  not reach. SpinQuant learns the rotation instead of fixing it; DeepSeek-V3.2's
  `rotate_activation` applies one to the head dim before fp8 quantisation.

  Q has to be cheap, because unlike the weight side the activation rotation runs
  every token. A Hadamard matrix is the choice: entries are +-1 so there are no
  multiplies at all, and it factors into log2(D) stages of pairwise add/subtract,
  so the cost is D log D instead of D^2. At D = 128 that is 7 stages of 64 pairs.

THIS IS A COMPUTE SHAPE NOTHING ELSE IN THE SUITE HAS
  Every output depends on every input, which is what a GEMM does -- but there is
  no contraction, no accumulator, and no operand pair. It is not a reduction
  either: a reduction collapses an axis and this preserves it, D in and D out.
  And it is not an elementwise chain, because the partner an element combines
  with MOVES between stages, doubling its distance each time.

    gemm            [T,D] x [D,D] -> [T,D],  D multiply-accumulates per output
    reduce          [T,D] -> [T],            one fold, axis destroyed
    elementwise     lane i touches lane i, forever
    butterfly       lane i touches lane i^h, for h = 1, 2, 4, ... -- log D
                    stages, each a full-width add and a full-width subtract,
                    with a DIFFERENT permutation between every pair of them

  What it asks for is a data movement of stride h WITHIN the vector axis, log D
  times with h doubling. On a lane machine each stage is a shuffle followed by an
  add and a sub; a backend with no intra-vector shuffle has to go through memory
  every stage, which turns 7 register operations into 7 round trips and is
  exactly the thing worth knowing.

HOW THE STAGES ARE WRITTEN, AND WHY IT HAD TO BE THIS WAY
  The stage is its own `@triton.jit` helper taking its geometry as `tl.constexpr`
  PARAMETERS, called four times with literal arguments -- the same construction
  sort_bitonic.py arrived at, for the same reasons. A `while` over the stride is
  a RUNTIME loop in triton, so the stride becomes a tensor and `tl.reshape`
  refuses it as a shape element; a jit body cannot read a module global; and a
  constexpr parameter used directly IS accepted as a shape element. So the
  log-depth recursion is unrolled at the call site and the recursion's geometry
  travels in the signature.

  Pairing without a gather: view [T, D] as [T, g, 2, h] where g = D/(2h). The
  element at (a, b, c) has flat index a*2h + b*h + c, so bit h of the index IS b
  -- the two partners of a butterfly are exactly b = 0 and b = 1 with a and c
  fixed. Permuting the 2 to the end and `tl.split` peels them apart as two
  registers; `tl.join` puts the sum and the difference back. No address
  arithmetic and no memory traffic, which is the whole point of a butterfly.

EXACTNESS. The normalisation is 1/sqrt(D) = 1/4 at D = 16, a power of two, so it
is exact in fp32 and folded into the last stage rather than costing a pass. Every
other operation is an add or a subtract. The `reference` here performs the same
four stages in the same order, so it agrees to the last bit; the independent
check is against an explicit Sylvester Hadamard matrix in float64, which is the
definition this is supposed to implement.

D = 16 gives four stages, small enough that all four calls fit on one screen.
Real QuaRot uses D = head_dim = 128, i.e. seven, and for a non-power-of-two
hidden size it splits into a power-of-two Hadamard times a small hard-coded one.

GREEN SINCE 2026-09-04, max_abs_err 0. `tl.split` on the partner axis becomes a
RANK-REDUCING `memref.subview` -- `[:, :, 0]` of a [16, 8, 2] tile -- and the
view it leaves is `memref<16x8xf32, strided<[16, 2]>>`, whose minor axis steps
TWO. vectorize emitted a `vector<8xf32>` load on it anyway and stage 5's
verifier said no: `'vector.load' op most minor memref dim must have unit stride`.

It asked the wrong thing. `_is_unit_step_in_bank` reads the strides a lane's
BANK has, and `spad_strides` derives those through `expand_shape` and otherwise
falls back to the convention -- which says the minor axis steps one, because a
subview's own strides are in its TYPE and nothing was reading them. The width of
a load is not a banking question; it is what the type says, so the type is now
asked directly. Where the answer is no the nest goes scalar, which is slower and
right. Note the same kernel already had the good form beside the bad one --
`memref<16x2x8xf32>` sliced to `strided<[16, 1]>` -- so only the stages whose
partner axis ends up MINOR were ever affected.

QuaRot/SpinQuant: a log-depth orthogonal butterfly. Not a GEMM (no contraction), not a reduction (the axis survives), not elementwise (the partner doubles its distance each stage). Adds and subs only, and a power-of-two normalisation, so it should be near-exact.

LAYOUT PIN. `expect_ir` holds `torchsim.copy` in 04-passes/04-torchsim-materialize-layouts.mlir at 0; what that pins is the conversion's one-axis default on the FIRST non-unit axis; the last axis fails the reshape verifier.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, D = 16, 16          # tokens, feature dim -- log2(16) = 4 butterfly stages
INV = 0.25             # 1/sqrt(D); a power of two, so exact in fp32


def kernel():
    import triton
    import triton.language as tl

    # ONE BUTTERFLY STAGE. `g`, `h` and the scale are constexpr PARAMETERS, so
    # the recursion's geometry is in the signature and the caller supplies
    # literals -- see the module docstring for why nothing else works.
    @triton.jit
    def _bfly(x, T: tl.constexpr, g: tl.constexpr, h: tl.constexpr,
              sc: tl.constexpr):
        # [T, D] -> [T, g, 2, h]: bit `h` of the flat index becomes axis 2.
        y = tl.reshape(x, [T, g, 2, h])
        y = tl.permute(y, (0, 1, 3, 2))          # partners to the last axis
        lo, hi = tl.split(y)                     # index bit clear / set
        y = tl.join((lo + hi) * sc, (lo - hi) * sc)
        y = tl.permute(y, (0, 1, 3, 2))
        return tl.reshape(y, [T, g * 2 * h])

    @triton.jit
    def hadamard_rotate_kernel(x_ptr, o_ptr, T: tl.constexpr, D: tl.constexpr,
                               INV: tl.constexpr):
        rt = tl.arange(0, T)
        rd = tl.arange(0, D)
        off = rt[:, None] * D + rd[None, :]

        x = tl.load(x_ptr + off)
        # The distance between partners DOUBLES every stage: 1, 2, 4, 8.
        x = _bfly(x, T, 8, 1, 1.0)
        x = _bfly(x, T, 4, 2, 1.0)
        x = _bfly(x, T, 2, 4, 1.0)
        x = _bfly(x, T, 1, 8, INV)      # 1/sqrt(D) folded into the last stage
        tl.store(o_ptr + off, x)

    return hadamard_rotate_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # An activation with the per-channel outliers the rotation exists to kill:
    # two feature dims at 30x, identically in every token.
    x = torch.randn(T, D, generator=g, dtype=torch.float32)
    heavy = torch.zeros(1, D)
    heavy[0, 3] = heavy[0, 11] = 1.0
    return {"x": x * (1.0 + 29.0 * heavy)}


def reference(inputs):
    import torch
    # The same four stages in the same order, so this is bit-identical to the
    # kernel rather than an approximation of it.
    x = inputs["x"].reshape(T, D).clone()
    h = 1
    while h < D:
        y = x.reshape(T, D // (2 * h), 2, h)
        lo, hi = y[:, :, 0, :], y[:, :, 1, :]
        sc = INV if 2 * h == D else 1.0
        x = torch.stack([(lo + hi) * sc, (lo - hi) * sc], dim=2).reshape(T, D)
        h *= 2
    return {"o": x.contiguous()}


SPEC = KernelSpec(
    name="hadamard_rotate_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "T": "constexpr",
               "D": "constexpr", "INV": "constexpr"},
    constexprs={"T": T, "D": D, "INV": INV},
    args=[Arg("x", IN, "float32", (T, D)),
          Arg("o", OUT, "float32", (T, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
    # The butterfly's half written back into its tile is ONE permute that writes the
    # window's rows and lanes and leaves the rest -- no DRAM trip.
    expect_ir={"04-passes/04-torchsim-materialize-layouts.mlir": {"torchsim.copy": 0},
               "04-transformed.mlir": {"dram_arg = -1": 0}},
)
