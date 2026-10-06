"""A side-by-side wraparound: the COLUMN index carries the `% WM`.

Mirror of ptr_wrap_stacked_run.py with the modulo moved to the other axis, which
in the reference lowering picks an entirely different arm (createSideBySideCastOps
rather than createStackedCastOps). Here it changes exactly one thing: which dim
the index generic reads with linalg.index, and therefore whether that index is
the lane number or the position inside a lane's vector.

minimal stacked wrap
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
WM, BM, BN = 12, 16, 32          # 12 < 16 -> the seam falls inside the tile
NX = 64 * BN
def kernel():
    import triton, triton.language as tl
    @triton.jit
    def ptr_wrap_side_by_side_run_kernel(x_ptr, o_ptr, WM: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM); rn = tl.arange(0, BN)
        cols = rn % WM
        v = tl.load(x_ptr + rm[:, None] * BN + cols[None, :])
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], v)
    return ptr_wrap_side_by_side_run_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}
def reference(inputs):
    import torch
    rm = torch.arange(BM); rn = torch.arange(BN)
    return {"o": inputs["x"][rm[:, None] * BN + (rn % WM)[None, :]].reshape(-1)}
SPEC = KernelSpec(name="ptr_wrap_side_by_side_run_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "WM": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"WM": WM, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, )
