"""A gather on the lane axis at two elements per lane, with the tile dumped.

    b       [16, 256] holding its own COLUMN NUMBER in every element
    load    b_ptr + offs_k[:, None] * N + (arange(BN) % N)[None, :]
    staged  that tile, stored out verbatim
    c       and the product, so the matmul still demands N on the lanes

WHY THE TILE IS STORED. The defect this pins was chased for three rounds
through a matmul's output, and the output could not tell two different things
apart: a gather that fetches the wrong element and a store that never writes
the slot look identical when the buffer starts at zero. Storing the STAGED TILE
names, for every destination slot, which source element landed in it -- because
b holds its own column number. That is rule 12's third layer done on purpose
instead of by inference.

WHAT IT CAUGHT. Before b09_vectorize added the per-lane slot to the lane
index, row 0 of the staged tile read

    cols 0..127     0, 1, ..., 127     right
    cols 128..255   0, 1, ..., 127     the same values AGAIN

so the destination advances through all 256 columns perfectly well -- every
slot was written -- and it is the INDEX that only ever spanned one element per
lane. `pytorchsim.lane_id` answers "which lane", which is the whole position only
at depth 1. That distinction is what the product could not show, and it is why
an earlier attempt to fix this from the output side went the wrong way.

After: 256 distinct values, 0..255, in the right slots, max_abs_err 0.

BN = 256 ON 128 LANES IS THE POINT. At BN = 128 a lane owns one slot and the
defect has nowhere to show. The matmul is not decoration either: without one the
tile banks on the axis the row runs along and the gather never lands on the lane
axis at all.

Stores the gathered tile itself, so every destination slot names the source element that landed in it. Reads 0..127 twice instead of 0..255 without the per-lane slot in the lane index.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 16, 256, 16
BM, BN, BK = 16, 256, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gather_on_lane_axis_deeper_kernel(a_ptr, b_ptr, staged_ptr, c_ptr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN) % N          # the wrap -> indirect
        offs_k = tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b)                      # demands N on the lanes
        tl.store(staged_ptr + offs_k[:, None] * N + tl.arange(0, BN)[None, :], b)
        tl.store(c_ptr + offs_m[:, None] * N + tl.arange(0, BN)[None, :], acc)

    return tile_gather_on_lane_axis_deeper_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # b holds its own column number, so the staged tile names where each
    # element came from.
    b = torch.arange(N, dtype=torch.float32).repeat(K, 1)
    return {"a": torch.randn(M, K, generator=g, dtype=torch.float32), "b": b}


def reference(inputs):
    import torch
    return {"staged": inputs["b"].clone(),
            "c": torch.matmul(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="tile_gather_on_lane_axis_deeper_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "staged_ptr": "*fp32",
               "c_ptr": "*fp32", "M": "constexpr", "N": "constexpr",
               "K": "constexpr", "BM": "constexpr", "BN": "constexpr",
               "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("staged", OUT, "float32", (K, N)), Arg("c", OUT, "float32", (M, N))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
