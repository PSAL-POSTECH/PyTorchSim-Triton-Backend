"""A store whose mask trims the tail writes the window at the wrong end.

    keep = rm < VALID                 VALID rows of a BM-row tile
    o[rm, rn] = x[rm, rn] + 1         stored under `keep`

The tile is BM rows and only the first VALID of them are real, so the mask takes
the LAST BM - VALID away. What lands in the buffer is the window moved by that
amount instead: the first BM - VALID rows are never written and the rest are
exactly right, values and positions both.

MEASURED on resnet18's sixteenth conv, whose 7x7x512 output is 49 rows of a
BLOCK_M = 64 tile:

    output rows 0..14   never written      (15 = 64 - 49)
    output rows 15..48  correct to 1e-4

Nothing else about that kernel is wrong -- the arithmetic, the layout and the
channel blocking all check out, and 34 of its 49 rows are exact. resnet18's
FIRST conv does not reach it because 12544 divides by its BLOCK_M, so the mask
trims nothing and the window has nowhere to move.

WHY THE SECOND AXIS IS HERE. With a single axis the store is one contiguous run
and its origin is the buffer's, so a window that starts late has nothing to
start late relative to. The row stride is what turns the offset into a gap.

AND IT IS ALSO WHY THE STORE USED TO BE A tts.scatter, which is the second thing
this file pins. `rm + VALID*rn` reaches both axes, so the `%` read as a wrap on
the SUM and the `+ rn` after it then had nowhere to go. Two facts lift it to one
strided pointer with `static_shape = [VALID*N, 0]`, and neither is about this
kernel:

    the term is DEAD in the `%`     VALID*rn is a multiple of VALID, so
                                    (rm + VALID*rn) % VALID IS rm % VALID, and
                                    the axis carrying it is left standing still
                                    -- visitOperandRemAsWrap, beside the two
                                    facts below rather than a stage above
    the axis has to CONTRIBUTE      what is left is a broadcast, stride 0 on
                                    axis 1 -- PtrAnalysis counted axis 1 as
                                    "wide" by its EXTENT, and an axis that does
                                    not move adds nothing the modulo can act on,
                                    exactly as a size-1 axis does not

A masked store whose tile is longer than the valid extent writes its window BM - VALID rows late, leaving that many rows unwritten.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

VALID, BM, N = 49, 64, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def masked_store_short_tile_kernel(x_ptr, o_ptr, VALID: tl.constexpr,
                                       BM: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, N)
        keep = rm < VALID
        v = tl.load(x_ptr + rm[:, None] * N + rn[None, :],
                    mask=keep[:, None], other=0.0)
        # THE MODULO ONLY LOOKS LIKE IT SPANS BOTH AXES. `VALID * rn` is a
        # multiple of VALID, so it is whole laps inside the `%` -- and what is
        # left reaches only axis 0, so the wrap is per-axis and the store is a
        # tts.store on a plain strided pointer rather than a tts.scatter.
        flat = rm[:, None] + VALID * rn[None, :]
        row = flat % VALID
        # THE WRAP IS WHAT MAKES THE MASK OBSERVABLE. rm >= VALID folds onto
        # 0..BM-VALID-1, so a store that ignores `keep` does not run off the end
        # -- it lands on rows that are already right and overwrites them with
        # the zeros `other` gave the invalid rows.
        tl.store(o_ptr + row * N + rn[None, :], v, mask=keep[:, None])

    return masked_store_short_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(VALID * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(VALID, N).reshape(-1)}


SPEC = KernelSpec(
    name="masked_store_short_tile_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "VALID": "constexpr",
               "BM": "constexpr", "N": "constexpr"},
    constexprs={"VALID": VALID, "BM": BM, "N": N},
    args=[Arg("x", IN, "float32", (VALID * N,)),
          Arg("o", OUT, "float32", (VALID * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE NUMBER CANNOT SAY IT: this answered 0 as a tts.scatter too. The rung
    # is the claim -- one address for the tile, not one per element.
    expect_ir={"07-regularized.mlir": {"tts.scatter": 0, "tts.gather": 0}},
)
