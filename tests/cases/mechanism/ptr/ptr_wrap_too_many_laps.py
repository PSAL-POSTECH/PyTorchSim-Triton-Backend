"""Sixteen laps: past MAX_WRAP_PIECES, so the wrap goes back to the index.

The threshold's REFUSE side. ptr_wrap_three_laps.py is the accept side, and a
threshold with only one side pinned drifts in whichever direction nobody looks.

WHY THERE IS A THRESHOLD AT ALL, measured 2026-09-04 by compiling one shape at
a series of lap counts with MAX_WRAP_PIECES forced high and forced to 1:

    laps        2     4     8    11    13    16    32    64
    rank 1     --    18    30    --    --    55   104   203   instructions, pieces
    rank 1     --    46    46    --    --    46    46    51   instructions, indirect
    rank 2     13    21    37    50    57    69    --    --   instructions, pieces
    rank 2     66    66    66    67    67    66    --    --   instructions, indirect

Two facts fall out and they are the whole design:

  ONE INDIRECT TRANSFER IS A CONSTANT. It does not grow with the lap count, and
  it does not grow with the TILE either -- widening the rank-2 tile from 32 to
  128 columns moved it 63 -> 66, because the address tile is built with vector
  ops and a wider tile is longer vectors, not more instructions.

  A PIECE COSTS ~3.1 INSTRUCTIONS, flat. Piece SIZE does not matter: 256
  elements in 8 runs of 32 costs the same as 64 in 8 runs of 8 (+17 vs +16).
  Only the COUNT does.

So the crossover is C_indirect / C_piece: 13 at rank 1, 15 at rank 2, and
higher again at rank 3 where the constant is bigger. 12 sits under all of them,
which costs a win of at most ~10 instructions at rank 2 and never takes a
losing trade. The old value was 16 -- inside the losing range at rank 1, which
is where ptr_modulus_multiples_arms sat: it emitted 23 descriptors and 124
instructions where the index does it in 5 and 70.

N = 128 over W = 8 is 16 laps: one past the accept side by enough that a small
change in the constant does not silently flip this file.

16 laps, past MAX_WRAP_PIECES -> the index carries the wrap. The refuse side of the threshold; ptr_wrap_three_laps.py is accept.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
W = 8            # 128 / 8 = 16 laps, over MAX_WRAP_PIECES


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_too_many_laps_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                      W: tl.constexpr):
        t = tl.arange(0, N)
        tl.store(o_ptr + t, tl.load(x_ptr + (t % W)))

    return ptr_wrap_too_many_laps_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"][torch.arange(N) % W]}


SPEC = KernelSpec(
    name="ptr_wrap_too_many_laps_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "W": "constexpr"},
    constexprs={"N": N, "W": W},
    args=[Arg("x", IN, "float32", (W,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-transformed.mlir": {"cut": 0, "offset_stride": 1,
                                   "pytorchsim.transfer": 2}},
)
