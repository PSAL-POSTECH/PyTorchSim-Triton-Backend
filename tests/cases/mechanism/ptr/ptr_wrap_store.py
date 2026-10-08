"""A wraparound on the STORE side.

The linalg lowering's StoreConverter has no wrap arm at all (census
store-no-wraparound-handling: it hands the two-operand cast to
materialize_in_destination as if it were one memref), so this is a shape the
reference lowering cannot emit. Reducing wrap to an indirect transfer makes the
direction irrelevant: same index, only dma_kind differs.

The reference below writes rows in order because later rows overwrite earlier
ones at the same wrapped destination -- that aliasing is the point of the shape.

wrap on store
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
WM, BM, BN = 12, 16, 32
NO = 64 * BN
def kernel():
    import triton, triton.language as tl
    @triton.jit
    def ptr_wrap_store_kernel(x_ptr, o_ptr, WM: tl.constexpr,
                                BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM); rn = tl.arange(0, BN)
        v = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        tl.store(o_ptr + (rm % WM)[:, None] * BN + rn[None, :], v)
    return ptr_wrap_store_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM * BN, generator=g, dtype=torch.float32)}
def reference(inputs):
    import torch
    o = torch.zeros(NO, dtype=torch.float32)
    x = inputs["x"].view(BM, BN)
    for r in range(BM):                    # later rows overwrite earlier ones
        o[(r % WM) * BN:(r % WM) * BN + BN] = x[r]
    return {"o": o}
SPEC = KernelSpec(name="ptr_wrap_store_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "WM": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"WM": WM, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (BM * BN,)), Arg("o", OUT, "float32", (NO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, )
