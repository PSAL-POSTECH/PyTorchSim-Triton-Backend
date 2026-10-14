"""ops_trans_matmul_maps, but with the second matmul's K past one K pair of the matrix unit.

WHY THIS SHAPE AND NOT THE OTHER ONE. `ops_trans_matmul_maps` puts non-default
`indexing_maps` on a matmul -- absorb_layout folds the `tl.trans` in by WRITING
them, so the op says "my A is (m,n,k) -> (k,m)" over a buffer that is physically
[K, M]. What it cannot show is whether lower_to_vcix's ADDRESS arithmetic reads
those maps, because every term that would expose it is multiplied by a loop
index that is zero there:

    sp = k_idx * M + m_idx * push_length + i        the input push's in-bank offset

`k_idx` is the trip counter of the K loop, and that loop only exists when
K > sysK. In ops_trans_matmul_maps K is 16 against a 256-deep array, so k_idx is
always 0 and `k_idx * M` never has to be right. Here the second matmul's K is
M1 = 512 against the same array, so the loop runs twice and the second trip has
to land on the right half of every lane's bank.

    p @ q     = a            [512, 16] -- M1 is the one that matters
    a.t()     = at           [16, 512] -- absorbed into the matmul's maps
    at @ r    = o            [16, 16],  contracting over 512 = 2 K pairs

WHAT IT IS ASKING. Whether `k_idx * M` is true BECAUSE A is [M, K] -- in which
case a transposed A breaks it -- or because A's LANE AXIS is K whatever the axis
order, which is what the axis graph's MUST pins and what the DMA's outer loop
stride follows. Those two readings agree on every kernel in the suite until K
crosses a tile with the maps non-default, which is this file and nothing else.

THE ANSWER, MEASURED 2026-09-04: the second reading. It passes, and it passed on
the tree before the bank-bound fix too, so it pins nothing about that. What it
does pin is `_matmul_mnk`: `k_idx * M` is right because A's LANE AXIS is K -- the
axis graph's MUST pins that and the DMA's outer-loop stride follows it -- and not
because A is [M, K]. Verified by breaking it on purpose, which is the only way a
green kernel earns its place (rule 2):

    M_val = const_index(M)              this file 1.53e-04   ops_trans_matmul_maps 2.86e-06
    M_val = const_index(mtA.shape[0])   this file 255.048    ops_trans_matmul_maps 2.86e-06

Reading M off the axis order instead of the maps is caught HERE AND NOWHERE ELSE
in the suite, because k_idx is zero in every other kernel that has non-default
maps.

The IR it reaches, so the next reader can check it is still this shape:
    linalg.matmul indexing_maps = [(d0,d1,d2) -> (d2,d0), ...]
                  torchsim.lane_axis = array<i64: 0, 1, 1>
                  K = 512 (the systolic lowering's torchsim.systolic_tile is gone)

STILL OWED, AND NOT PINNED HERE. `_find_operand_transfers` takes `subtileM` off
the DMA descriptor as `sub[-2]` -- positional, and blind to the maps in a way
`_matmul_mnk` beside it is not. This kernel cannot discriminate it: subtileM
feeds `push_length = min(subtileM, sysM)` and sysM is 16 here, so 512 and 16
give the same answer. Tripping it needs a transposed A staged by a plain
transfer AND M past sysM.

torch equivalent: (p @ q).t() @ r.

Non-default matmul maps AND a K that crosses a K pair, which is the only shape where lower_to_vcix's `k_idx * M` term has to be right about which axis of A is which.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# M1 = 512 = 2 * the array's depth, so the second matmul's K loop runs twice.
# The others are 16, the smallest tl.dot accepts, to keep the buffers small.
M1, K1, N1, L = 512, 16, 16, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_trans_matmul_maps_deep_k_kernel(p_ptr, q_ptr, r_ptr, o_ptr,
                                            M1: tl.constexpr, K1: tl.constexpr,
                                            N1: tl.constexpr, L: tl.constexpr):
        rm = tl.arange(0, M1)
        rk = tl.arange(0, K1)
        rn = tl.arange(0, N1)
        rl = tl.arange(0, L)
        p = tl.load(p_ptr + rm[:, None] * K1 + rk[None, :])       # (M1, K1)
        q = tl.load(q_ptr + rk[:, None] * N1 + rn[None, :])       # (K1, N1)
        r = tl.load(r_ptr + rm[:, None] * L + rl[None, :])        # (M1, L)
        a = tl.dot(p, q)                                          # (M1, N1)
        at = tl.trans(a, 1, 0)                                    # (N1, M1)
        o = tl.dot(at, r)                                         # (N1, L)
        tl.store(o_ptr + rn[:, None] * L + rl[None, :], o)

    return ops_trans_matmul_maps_deep_k_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"p": torch.randn(M1 * K1, generator=g, dtype=torch.float32),
            "q": torch.randn(K1 * N1, generator=g, dtype=torch.float32),
            "r": torch.randn(M1 * L, generator=g, dtype=torch.float32)}


def reference(inputs):
    p = inputs["p"].reshape(M1, K1)
    q = inputs["q"].reshape(K1, N1)
    r = inputs["r"].reshape(M1, L)
    return {"o": ((p @ q).t() @ r).reshape(-1)}


SPEC = KernelSpec(
    name="ops_trans_matmul_maps_deep_k_kernel", kernel=kernel,
    signature={"p_ptr": "*fp32", "q_ptr": "*fp32", "r_ptr": "*fp32",
               "o_ptr": "*fp32", "M1": "constexpr", "K1": "constexpr",
               "N1": "constexpr", "L": "constexpr"},
    constexprs={"M1": M1, "K1": K1, "N1": N1, "L": L},
    args=[Arg("p", IN, "float32", (M1 * K1,)), Arg("q", IN, "float32", (K1 * N1,)),
          Arg("r", IN, "float32", (M1 * L,)), Arg("o", OUT, "float32", (N1 * L,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # 512 products per output element, summed in the array's order and not
    # torch's, so this is the matmul tolerance rather than the elementwise one.
    rtol=1e-3, atol=1e-3,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
