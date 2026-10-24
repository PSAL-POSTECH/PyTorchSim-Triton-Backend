"""A matmul whose A tile is written in the scratchpad before it is streamed --
and the SCALE the elementwise op reads, whose lower half was never filled.

    x = tl.load(...)              [32, 16]
    tl.dot(x, w)                  -> max_abs_err 1.90735e-06, green
    tl.dot(x * 2.0, w)            -> max_abs_err 36.9027, rows 16..31 come back
                                     EXACTLY 0.0, and nothing refuses it

WHAT IT PINS, NOW THAT IT IS DIAGNOSED. `x * 2.0` is not one op. The linalg lowering
materialises the 2.0 as a TILE, so the IR is a `linalg.fill` of a
`memref<32x16xf32, 1>` followed by the multiply, and that tile is banked on its
LAST axis because everything in this group is:

    linalg.fill {pytorchsim.lane_axis = array<i64: -1, 1>} ins(%cst_0 : f32)
                                                     outs(%3 : memref<32x16xf32, 1>)

Slot 0 of that array is the SCALAR's, slot 1 is the buffer's, and
vectorize's fill path read slot 0 -- see `_lane_axes` in
p05_vectorize, where the misalignment and its 555-op census are written
down. Read as "no axis" the fill fell back to iteration dim 0 and emitted one
`vector<16>` per lane, so 16 of every lane's 32 in-bank slots got 2.0 and the
other 16 kept the zero the `.spad` array loads with. A's rows 16..31 were
therefore `x * 0`, and an all-zero A row produces an all-zero C row.

    read back off spike's own runtime/o.raw, either side of the fix:
      rows  0..15   BYTE-IDENTICAL
      rows 16..31   exactly 0.0 before, the reference to 3.8147e-06 after

That also explains the size table this file was collecting, which is the part
that should have given it away:

    M = 32, K = 16, N = 32   rows 0..15 exact, rows 16..31 exactly 0.0
    M = 32, K = 32, N = 32   every row wrong (max_abs_err 62.6708)
    M = 16, K = 16, N = 32   green -- 3.8147e-06

M = 16, K = 16 makes the scale tile SQUARE, and a square tile is the one shape
where reading the lane axis off dim 0 and off the buffer's own stamp give the
same answer. It went green for that reason and not because 16 rows fit.

WHAT THIS FILE GOT WRONG, KEPT BECAUSE IT IS THE INSTRUCTIVE PART. It said "it
is not the lane axis being read wrong either -- the stamp is
`array<i64: 1, 1, 1>`". That stamp is the MATMUL's, and it was right about the
matmul. The fill three lines above it is a different op with a different array,
and no one opened it (rule 15a: a number read off the wrong sample).

WHAT WOULD BREAK IT. Reading a fill's lane axis out of any slot but the one
belonging to its outs buffer. Squaring the tile hides it, which is what
`M = K` does.

One elementwise op between the MVIN and the matmul used to zero the output rows past M/2, with no diagnostic: the scale it reads is a materialised tile, and the fill of a last-axis-banked tile was reading its lane axis out of the scalar operand's stamp slot.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 32, 16, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def a_written_in_sram_kernel(x_ptr, w_ptr, o_ptr, M: tl.constexpr,
                                 K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])       # [M, K]
        w = tl.load(w_ptr + rk[:, None] * N + rn[None, :])       # [K, N]
        # The scale is the entire kernel. Delete it and this goes green.
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(x * 2.0, w))

    return a_written_in_sram_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K, generator=g, dtype=torch.float32),
            "w": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, K)
    w = inputs["w"].reshape(K, N)
    return {"o": ((x * 2.0) @ w).reshape(-1)}


SPEC = KernelSpec(
    name="a_written_in_sram_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M * K,)), Arg("w", IN, "float32", (K * N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
