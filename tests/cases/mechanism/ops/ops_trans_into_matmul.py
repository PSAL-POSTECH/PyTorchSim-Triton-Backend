"""A transpose no READER can take, absorbed BACKWARD into the MVIN that staged it.

WHAT IT PINS. absorb_layout_ops's backward arm, `_plan_back`. The transpose's only
consumer is a `linalg.matmul`, whose maps ARE its definition -- (m, n, k) over
A(m, k) B(k, n) C(m, n) -- so there is no attribute a permutation could be
composed into and the forward walk correctly refuses. Backward it is free: the
transposed value's source has one producer chain, `bufferization.to_tensor` of a
`memref.alloc` that a single MVIN fills, and the whole permutation is that
transfer's `dram_stride`. The DMA reads the tile transposed and no element ever
moves between lanes.

WHY THIS IS A SEPARATE WALK AND NOT THE FORWARD ONE RUN THE OTHER WAY. `_absorb`
used to try the backward direction by handing the FORWARD planner the producer
and the index of its last operand. On a `to_tensor` that index is 0, the THROUGH
branch fires, and that branch retypes the op's own RESULT and recurses into THAT
value's readers -- so the walk turns around and asks the transpose being
absorbed, which refuses itself. Measured before the fix: `_plan_into` came back
`refused=['linalg.transpose']`.

AND IT HAS TO REACH THE BUFFER, not only the ops. The MVIN produces no SSA value
at all: it is another USER of the `memref.alloc` the `to_tensor` reads. A walk
that follows definitions and stops there retypes every op on the chain and
leaves the alloc -- and the transfer filling it -- typed for the layout that is
gone, which is rule 4's most common half-rewrite.

WHAT WOULD BREAK IT. Compose the permutation the wrong way round in the backward
direction -- `perm` where the source axes want `_invert(perm)` -- and the DMA
reads the wrong image: this is 32x8 into 8x32, not self-inverse in the SHAPE
even though [1, 0] is self-inverse as a permutation, so the transfer's tile and
its `dram_stride` stop agreeing and spike answers wrong numbers. Retype the ops
on the chain and leave `memref.alloc` alone and the module stops verifying at
the transfer. Restore the old single-use gate on the SOURCE and this still
passes -- the gate is not what this file is for; it was removed because
`_plan_back` asks every other reader of every value it turns, which is the thing
the gate stood in for.

WHAT IT DOES NOT PIN, SAID OUT LOUD (the same admission ops_relayout_shapes
makes). The VERDICT does not move: without the backward arm the transpose
survives absorb_layout_ops, fit_layout_to_lanes turns it into a DRAM round trip, and
the kernel still answers correctly. What moves is the COST, and a KernelSpec has
no round-trip budget to assert it with -- so the evidence is regress.py's own
column, which prints `[1 round trip]` for this file without the backward arm and
nothing at all with it. Rule 8b makes that number the metric; this file is where
it is read.

torch equivalent: x.t() @ y.

A permutation the consumer cannot take is still free at the producer: one dram_stride on the MVIN that staged the tile.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 32, 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_trans_into_matmul_kernel(x_ptr, y_ptr, o_ptr, M: tl.constexpr,
                                     K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])       # (M, K)
        y = tl.load(y_ptr + rm[:, None] * N + rn[None, :])       # (M, N)
        xt = tl.trans(x, 1, 0)                                    # (K, M)
        o = tl.dot(xt, y)                                         # (K, N)
        tl.store(o_ptr + rk[:, None] * N + rn[None, :], o)

    return ops_trans_into_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K, generator=g, dtype=torch.float32),
            "y": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, K)
    y = inputs["y"].reshape(M, N)
    return {"o": (x.t() @ y).reshape(-1)}


SPEC = KernelSpec(
    name="ops_trans_into_matmul_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M * K,)), Arg("y", IN, "float32", (M * N,)),
          Arg("o", OUT, "float32", (K * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
