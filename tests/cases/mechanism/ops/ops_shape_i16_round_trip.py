"""The same shuffles as ops_shape, carried at i16: the emitter asks a byte question.

SIBLING: ops_shape_i64_round_trip.py is this file at i64 and carries the long
explanation of why a shuffle mints a DRAM round trip at all, and of the three
conditions a witness for one has to satisfy. This changes the element type and
nothing else, for the OTHER half of the width rule.

WHAT IT PINS. Two questions were being answered by one predicate:

    _scratch_carries       should this round trip be CHOSEN?   >= 32 bits
    _addresses_as_scratch  can the chosen trip be BUILT?       a whole byte

The first belongs to the minters: a narrow row that reaches one is better
widened (_travels_as_scratch) than carried, and four green kernels depend on
that. The second is all the emitter has left to ask -- three of the four minters
never asked about the width at all, so what arrives there is not a choice to
reconsider. It can be built whenever the descriptor's element_size is the
element's real size, which is every width that is a whole number of bytes:
lower_to_dma takes element_size from the SRAM tile and _size_scratch
sizes the buffer in those same bytes.

Sub-byte is the case the old refusal was really about, and it still refuses:
`_elem_bytes` rounds one bit up to one byte because a descriptor has no smaller
unit, while the vector unit bit-packs a mask -- 16 elements in two bytes. The
measurement is at SCRATCH_BITS (mask_col_flag_broadcast_over_rows answering
2.56723, one column of eight arriving).

    develop f6232e1   Fatal: a DRAM round trip of i16 cannot describe its own
                      elements
    with the fix      ALL STAGES PASSED, max_abs_err 0

FOUND BY DeepSeek-V3's top-k, which needs BOTH widths. Inductor narrows the sort
index to i16 -- there are 32 elements to sort, so it fits -- and widens it back
to i64 after, so the same kernel round-trips an i64 tile and an i16 one. Fixing
only the 64-bit half moved the refusal from one to the other.

torch equivalent: ops_shape's, in int16 (see reference).

ops_shape's shuffles at i16: pins the emitter asking whether a trip can be built (a whole byte) rather than whether to choose it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 4           # outer-product side
K = 8           # vector length
XN = 64
SN = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_shape_i16_round_trip_kernel(x_ptr, s_ptr, M: tl.constexpr,
                                        K: tl.constexpr):
        rows = tl.arange(0, M)
        cols = tl.arange(0, K)
        a = tl.load(x_ptr + cols)
        b = tl.load(x_ptr + K + cols)

        j = tl.join(a, b)                              # (K, 2)
        c = tl.cat(a + 1, b - 1, can_reorder=True)     # (2K,)
        mrs = tl.reshape(c * 2, (2, K))                # 1-D -> 2-D  EXPAND
        t = tl.trans(mrs + 1, 1, 0)                    # (2,K) -> (K,2) TRANS
        d = tl.reshape(t * 3, (2, K))                  # (K,2) -> (2,K) DYNAMIC
        f = tl.reshape(d + 4, (2 * K,))                # 2-D -> 1-D  COLLAPSE
        tl.store(s_ptr + tl.arange(0, 2 * K), f)

        p, r = tl.split(j * 2 + t)                     # (K,2) -> two (K,)
        tl.store(s_ptr + 2 * K + cols, p - r)

        ar = tl.load(x_ptr + rows)
        bc = tl.load(x_ptr + M + rows)
        outer = ar[:, None] * bc[None, :]              # (M,1) x (1,M) -> (M,M)
        tl.store(s_ptr + 3 * K + rows[:, None] * M + rows[None, :], outer)

    return ops_shape_i16_round_trip_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-100, 100, (XN,), generator=g,
                               dtype=torch.int16)}


def reference(inputs):
    import torch
    x = inputs["x"]
    out = torch.zeros(SN, dtype=torch.int16)
    a, b = x[:K], x[K:2 * K]
    j = torch.stack([a, b], dim=1)
    c = torch.cat([a + 1, b - 1])
    mrs = (c * 2).reshape(2, K)
    t = (mrs + 1).t()
    d = (t * 3).reshape(2, K)
    out[:2 * K] = (d + 4).reshape(2 * K)
    pr = j * 2 + t
    out[2 * K:3 * K] = pr[:, 0] - pr[:, 1]
    ar, bc = x[:M], x[M:2 * M]
    out[3 * K:3 * K + M * M] = (ar[:, None] * bc[None, :]).reshape(-1)
    return {"s": out}


SPEC = KernelSpec(
    name="ops_shape_i16_round_trip_kernel", kernel=kernel,
    signature={"x_ptr": "*i16", "s_ptr": "*i16",
               "M": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "K": K},
    args=[Arg("x", IN, "int16", (XN,)), Arg("s", OUT, "int16", (SN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
