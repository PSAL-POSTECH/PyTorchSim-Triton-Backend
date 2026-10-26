"""`tl.arange` on the MIDDLE axis of a rank-3 tile -- neither lanes nor vector."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 8, 4, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_index_middle_dim_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                     K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)[:, None, None]
        rk = tl.arange(0, K)[None, :, None]
        rn = tl.arange(0, N)[None, None, :]
        idx = rm * (K * N) + rk * N + rn
        x = tl.load(x_ptr + idx)
        # rk is the MIDDLE axis: not the lane dim (that is rm) and not the
        # vector dim (that is rn). It survives as a linalg.index because its
        # VALUE is used, not just its address.
        tl.store(o_ptr + idx, x + rk.to(tl.float32))

    return tile_index_middle_dim_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, K, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    k = torch.arange(K, dtype=torch.float32).reshape(1, K, 1)
    return {"o": x + k}


SPEC = KernelSpec(
    name="tile_index_middle_dim_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M, K, N)),
          Arg("o", OUT, "float32", (M, K, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"08-passes/04-pytorchsim-materialize-layouts.mlir": {"pytorchsim.convert_layout": 0}},
)
