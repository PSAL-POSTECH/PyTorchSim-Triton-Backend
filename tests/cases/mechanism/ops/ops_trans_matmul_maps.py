"""A transpose between two matmuls -- red since 2026-08-26, and not for its title.

WHAT IT WAS WRITTEN FOR. `absorb_layout_ops`'s `linalg.matmul` arm: the `tl.trans`
between the two dots is absorbed by WRITING the second matmul's maps, and this
kernel reaches fit_layout_to_lanes carrying

    linalg.matmul indexing_maps = [(d0,d1,d2) -> (d2,d0),
                                   (d0,d1,d2) -> (d2,d1),     <- A is [k, m]
                                   (d0,d1,d2) -> (d0,d1)]

It never pinned that arm: it failed identically with the arm and with the arm
disabled, so by rule 2 it pinned nothing about it. It was kept for the number
the measurement found instead -- err 19.7735, compiled all the way to Spike with
nothing above the reference check saying a word (rule 12).

WHAT IT ACTUALLY PINNED, FIXED 2026-09-04. Not the maps. `lower_to_vcix` read
each operand with a `vector.transfer_read` of vlen/elen elements indexed
[sp floordiv row, sp mod row] on the TILE, `row = shape[-1]`. The pair is a
round trip -- the address flattens straight back to `sp` -- but the read's
bounds check is against that last axis, and the last axis is THE LANE COUNT.
A tile narrower than vlen/elen therefore had every read clamped there and the
rest padded with the pad value. Here `p` is [16, 8] over 8 lanes and 16 deep,
so the input push delivered 8 of its 16 rows and the other 8 arrived as zeros.

    old lower_to_vcix, vlen 512 (vlen/elen = 16 > 8)   err 19.7735
    old lower_to_vcix, vlen 128 (vlen/elen =  4 < 8)   err 2.86102e-06   <- the same IR

That is the whole of the old `expect_why` ("wrong at 256 lanes, right at 128, so
a lane-count assumption is loose"): right in kind, wrong in variable -- vlen sets
the vector's width, and the tile's last axis was being asked to bound it. The
fix reads through the `[1, in-bank]` view a lane's bank really has, whose depth
is lib_bank_layout's own `need_per_lane`, so the clamp now falls where the data
actually ends. The suite's round trips do not move: 95 before, 95 after, and this
kernel KEEPS its one relayout -- the sweep first said 94 because run.py's "GAP HAS
CLOSED" line dropped the trip note every other branch carries (rule 15a).

STILL OWED, AND STILL TRUE. `lower_to_vcix` reads `indexing_maps` for (M, N, K)
and NOWHERE ELSE -- zero occurrences in the file outside `_matmul_mnk` -- so its
address arithmetic still assumes A(m,k) B(k,n) C(m,n). This kernel no longer
demonstrates it: its non-default operand is A, banked on axis 0, and K here
equals one K tile, so the `k_idx * M` term that would expose it is zero. A
kernel that trips it needs non-default maps AND K past one tile.
Also still owed: a kernel that pins the matmul arm -- a transpose whose only
reader is a matmul and whose producer offers the backward walk nothing.

torch equivalent: (p @ q).t() @ r.

A matmul states its maps when they are not the default ones, so a permutation composes into it the way it composes into a generic. Green since the operand reads bound against the bank instead of the tile's last axis, which is the lane count (measured err 19.7735).
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N, L = 16, 8, 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_trans_matmul_maps_kernel(p_ptr, q_ptr, r_ptr, o_ptr,
                                     M: tl.constexpr, K: tl.constexpr,
                                     N: tl.constexpr, L: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        rl = tl.arange(0, L)
        p = tl.load(p_ptr + rm[:, None] * K + rk[None, :])        # (M, K)
        q = tl.load(q_ptr + rk[:, None] * N + rn[None, :])        # (K, N)
        r = tl.load(r_ptr + rm[:, None] * L + rl[None, :])        # (M, L)
        a = tl.dot(p, q)                                           # (M, N)
        at = tl.trans(a, 1, 0)                                     # (N, M)
        o = tl.dot(at, r)                                          # (N, L)
        tl.store(o_ptr + rn[:, None] * L + rl[None, :], o)

    return ops_trans_matmul_maps_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"p": torch.randn(M * K, generator=g, dtype=torch.float32),
            "q": torch.randn(K * N, generator=g, dtype=torch.float32),
            "r": torch.randn(M * L, generator=g, dtype=torch.float32)}


def reference(inputs):
    p = inputs["p"].reshape(M, K)
    q = inputs["q"].reshape(K, N)
    r = inputs["r"].reshape(M, L)
    return {"o": ((p @ q).t() @ r).reshape(-1)}


SPEC = KernelSpec(
    name="ops_trans_matmul_maps_kernel", kernel=kernel,
    signature={"p_ptr": "*fp32", "q_ptr": "*fp32", "r_ptr": "*fp32",
               "o_ptr": "*fp32", "M": "constexpr", "K": "constexpr",
               "N": "constexpr", "L": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "L": L},
    args=[Arg("p", IN, "float32", (M * K,)), Arg("q", IN, "float32", (K * N,)),
          Arg("r", IN, "float32", (M * L,)), Arg("o", OUT, "float32", (N * L,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect="blocked",
    expect_why="matmul K=8 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
