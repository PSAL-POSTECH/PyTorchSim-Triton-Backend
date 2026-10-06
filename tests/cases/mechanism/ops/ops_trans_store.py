"""A transpose whose only reader is the store, absorbed into the descriptor.

WHAT IT PINS. absorb_layout_ops's transfer sink: a `linalg.transpose` reaching a
`torchsim.transfer` through the `bufferization.to_buffer` that bufferization put
between them. The permutation goes into `dram_stride` and the op disappears, so
element (i, j) of the tile is WRITTEN at the offset the transposed image wants
and the DMA does the whole thing for free.

WHAT WOULD BREAK IT. Take the transfer out of `_plan_into`, or stop looking
through `to_buffer`, and the transpose survives absorb_layout_ops; vectorize
then refuses with "'linalg.transpose' still touches a scratchpad buffer".
Compose the permutation the wrong way round -- `ds[perm[a]]` where the source
axis wants `ds[perm.index(a)]` -- and this stays green, because [1, 0] is its
own inverse; that is what the rank-3 sibling below is for.

WHY NOT ops_relayout_shapes, WHICH ALSO TRANSPOSES. Its transpose has a second
reader, a `tensor.reshape`, whose two sides have no axis in common -- so the
absorption is refused there and the op survives on purpose. A sink is pinned by
a kernel where it is the ONLY reader.

torch equivalent: x.t().

The permutation lives in the MVOUT's dram_stride: input axis a takes the stride of the output axis the permutation sent it to, and no element moves between lanes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K = 8, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_trans_store_kernel(x_ptr, t_ptr, M: tl.constexpr, K: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])       # (M, K)
        t = tl.trans(x, 1, 0)                                     # (K, M)
        tl.store(t_ptr + rk[:, None] * M + rm[None, :], t)

    return ops_trans_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, K)
    return {"t": x.t().contiguous().reshape(-1)}


SPEC = KernelSpec(
    name="ops_trans_store_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "t_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "K": K},
    args=[Arg("x", IN, "float32", (M * K,)), Arg("t", OUT, "float32", (M * K,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
