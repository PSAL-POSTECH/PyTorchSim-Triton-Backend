"""A matmul accumulator SEEDED with a value, banked on its last axis, taller
inside a lane than it is wide across them -- and the half of the seed that was
never written.

WHAT IT PINS. vectorize reads `torchsim.lane_axis` per operand. The array is
stamped per OPERAND of the linalg op, and `linalg.fill` is the one linalg op
here whose first operand is not a buffer:

    linalg.fill {torchsim.lane_axis = array<i64: -1, 1>} ins(%cst : f32)
                                                     outs(%acc : memref<32x16xf32, 1>)

Slot 0 is the SCALAR's (-1, no axis); slot 1 is the accumulator's. The fill path
builds `memrefs` from `operands[1:]` and left the stamp array unshifted, so slot
0 was paired with the outs buffer, read as "no axis", and the lane dim fell back
to iteration dim 0. Every fill of a tile banked on its LAST axis was therefore
emitted as a tile banked on dim 0 -- the vector ran ALONG the lanes instead of
INSIDE a bank.

    [M, N] banked on axis 1, N lanes    a lane holds M elements
    the dim-0 reading issues            one vector<N> at bank offset 0

The two are the same number only for a square tile, which is why 41 of the 87
wrongly-read fills in the suite came out right anyway. Off square it goes wrong
in both directions, and this file is the direction a byte census cannot see:

    N > M   the vector runs off the end of the bank into the next global
    N < M   bank offsets N..M-1 are NEVER WRITTEN -- this kernel

WHAT MAKES IT VISIBLE HERE. An unfilled tail is invisible while the seed is
zero: the scratchpad globals are C arrays in `.spad` and start at zero, so
`tl.zeros` and "not written" agree. So the accumulator is seeded with a value
instead:

    acc = tl.full((BM, BN), SEED)      SEED != 0
    acc += tl.dot(a, b)                the array forces BN onto the lanes

With BM = 32 > BN = 16 each of the 16 lanes holds 32 elements -- one COLUMN of
the tile, its in-bank offset being the row. The dim-0 reading writes 16 of them,
so rows 0..15 get SEED and rows 16..31 get nothing, and every element of the
lower half of C comes back short by exactly SEED. Measured on the tree before
the fix: max_abs_err 3, 256 of the 512 elements outside tolerance, and no
diagnostic anywhere. Read back off spike's own `runtime/c.raw` either side of
the fix: rows 0..15 BYTE-IDENTICAL, rows 16..31 differing by exactly 3.0 and by
nothing else (`np.unique(after - before) == [0., 3.]`). After: 2.86102e-06.

WHY tl.dot AND WHY K = 2 * BK. The lane axis has to be the LAST one and the
matrix unit is the only thing that demands it -- an elementwise tile banks on
dim 0. And one K tile would let triton fold the reduction loop away, which is
where lower_to_vcix finds the accumulation. Both are borrowed from
mixed_gemm_partial_n; what this file is about is BM > BN with a non-zero seed.

WHAT WOULD BREAK IT. Reading the lane axis of the fill's OUTS out of slot 0 of
the stamp array -- or out of anywhere but the slot belonging to that buffer.
Making the tile square hides it again, and so does seeding with zero.

torch equivalent: c = SEED + a @ b

A last-axis-banked fill on a tile taller in a lane than it is wide across them. The fill has to cover the whole of each lane's bank; reading its lane axis out of the scalar operand's stamp slot leaves rows 16..31 of the seed unwritten (err 3.0).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# BM > BN is the point: a lane holds 32 and the fill's dim-0 reading writes 16.
M, N, K = 32, 16, 32
BM, BN, BK = 32, 16, 16
SEED = 3.0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_fill_lane_axis_tail_kernel(a_ptr, b_ptr, c_ptr,
                                        M: tl.constexpr, N: tl.constexpr,
                                        K: tl.constexpr, BM: tl.constexpr,
                                        BN: tl.constexpr, BK: tl.constexpr,
                                        SEED: tl.constexpr):
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        # NOT tl.zeros: a bank that was never written already reads zero, so a
        # zero seed makes the two indistinguishable.
        acc = tl.full((BM, BN), SEED, dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return tile_fill_lane_axis_tail_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    import torch
    return {"c": SEED + torch.matmul(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="tile_fill_lane_axis_tail_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr",
               "SEED": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK,
                "SEED": SEED},
    args=[
        Arg("a", IN, "float32", (M, K)),
        Arg("b", IN, "float32", (K, N)),
        Arg("c", OUT, "float32", (M, N)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # The matmul kernels' tolerance: accumulation order differs from torch's.
    rtol=1e-3, atol=1e-3,
)
