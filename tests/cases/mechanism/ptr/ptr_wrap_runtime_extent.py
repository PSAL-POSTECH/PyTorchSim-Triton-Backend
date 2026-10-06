"""A wraparound whose extent is a RUNTIME argument, not a constexpr.

`shape:` arrives as a kDynamic entry plus an SSA operand. The index generic
takes it the same way it takes a constant -- the modulus is just another value
in the expression -- so nothing about the lowering changes. What this pins down
is that the resolution path (the same one dynamic strides use) reaches it.

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
    def ptr_wrap_runtime_extent_kernel(x_ptr, o_ptr, WM,
                          BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM); rn = tl.arange(0, BN)
        rows = rm % WM
        v = tl.load(x_ptr + rows[:, None] * BN + rn[None, :])
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], v)
    return ptr_wrap_runtime_extent_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}
def reference(inputs):
    import torch
    rm = torch.arange(BM); rn = torch.arange(BN)
    return {"o": inputs["x"][(rm % WM)[:, None] * BN + rn[None, :]].reshape(-1)}
SPEC = KernelSpec(name="ptr_wrap_runtime_extent_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "WM": "i32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (BM * BN,))],
    extra={"scalar_args": [("WM", "int32_t")], "scalar_values": {"WM": WM}},
    grid=(1,), reference=reference, make_inputs=make_inputs, )
