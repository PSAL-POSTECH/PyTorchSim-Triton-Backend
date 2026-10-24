"""A wrapped STORE whose tile is deeper than one per lane on BOTH counts.

    tile   [256, 256] f32, lane_axis = 1, so 256 columns over 128 lanes
    store  c_ptr + (offs_m % M)[:, None] * N + (offs_n % N)[None, :]
    value  acc[m, c] == c, so every element names the column it belongs in

THE BOUNDARY, measured by counting DISTINCT values in the scatter's index
(layout-independent) and by reading the output:

    M = 16    4096 / 4096      PASS   err 0
    M = 128   32768 / 32768    PASS   err 0
    M = 256   32768 / 65536    FAIL   err 255      <- before
    M = 256   65536 / 65536    PASS   err 0        <- after

At M = 256 all 256 ROWS are present in the index and only 128 COLUMNS are, so
the output takes acc's columns 128..255 into its columns 0..127 and leaves
128..255 at zero. The row extent is what moves, and the columns are what break,
which is the shape of the answer: below the lane count the column index is a
`linalg.index` in a vectorised generic -- the path p05_vectorize now gives
the per-lane slot to -- and at 256 it is instead read back from a PREPARED
rank-1 row buffer, `memref<256xi32>` banked on axis 0 at depth 2, with
`pytorchsim.lane_splat`. AND THE WRITER AND THE READER OF THAT BUFFER DISAGREE ABOUT
HOW ITS 256 ELEMENTS ARE SPLIT OVER THE LANES:

    fill    affine.for %arg9 = 0 to 8 step 8 {        ONE trip, %arg9 = 0
              %11 = sf.vc.v.i.se(..., 8)              the lane id, vector<8>
              %14 = %arg9 * 128                       so always 0
              affine.vector_store %17, %3[0] : memref<256xi32, 1>, vector<8xi32>

    read    %10 = affine.vector_load %3[%arg9] {pytorchsim.lane_splat}
                    : memref<256xi32, 1>, vector<1xi32>     %arg9 in {0, 1}

AND THE FIX IS WHERE THE DEPTH LIVES. Read at vectorize's own output rather than five
passes later, the loop is

    affine.for %arg10 = 0 to 2 step 2 {          one trip, %arg10 = 0
      %8 = pytorchsim.vlane_idx : vector<2xindex>
      affine.vector_store %13, %3[%arg10] : memref<256xi32, 1>, vector<2xi32>

-- the STEP is the whole vector, so the depth is carried in the vector's own
positions and the loop counter never leaves 0. The counter term 313c242 added is
present and is multiplied by zero. When the lane axis is also the vector dim the
slot is `vector.step`, and adding that is what closes it: the index goes
32768 -> 65536 distinct and the answer to 0.

This is why the same defect needed two fixes and why reading the emitted RISC-V
loop (`0 to 8 step 8`, a vector of 8 after legalize widened it) named the
symptom and not the cause. vectorize's own output says `0 to 2 step 2`, which does.

WHAT IS NOT THE CAUSE, measured. Not the scatter itself: the same store wrapped
on both dims at M = 16 and M = 128 comes back exact, so an indirect MVOUT at two
elements per lane is fine. Not this session's lane-index fix either -- reverting
it leaves this identical, 32768 unique and the same output.

WHY THE VALUE NAMES ITS OWN COLUMN. `a` is one-hot on k = 0 and every row of `b`
holds its column number, so acc[m, c] == c by construction and any element in
the output says where it came from. Reading a defect like this out of an
ordinary matmul's numbers does not work: a scatter that skips a slot and a
gather that fetches the wrong element are indistinguishable when the buffer
starts at zero, which cost three rounds before the tile was made to name itself.

A wrapped store at 256 rows and 256 columns on 128 lanes. The scatter index carries all 256 rows and only 128 columns; M = 128 and M = 16 are both exact, so it is the prepared rank-1 row buffer the column index is read back from, not the scatter.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 256, 256, 16
BM, BN, BK = 256, 256, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_scatter_wrapped_rows_past_lanes_kernel(a_ptr, b_ptr, c_ptr,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b)                       # acc[m, c] == c
        # the WRAP is on the store only -> an indirect scatter
        tl.store(c_ptr + (offs_m % M)[:, None] * N + (offs_n % N)[None, :], acc)

    return tile_scatter_wrapped_rows_past_lanes_kernel


def make_inputs(torch, seed=0):
    a = torch.zeros(M, K, dtype=torch.float32); a[:, 0] = 1.0
    b = torch.arange(N, dtype=torch.float32).repeat(K, 1)
    return {"a": a, "b": b}


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="tile_scatter_wrapped_rows_past_lanes_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (M, K)), Arg("b", IN, "float32", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(1, 1), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
