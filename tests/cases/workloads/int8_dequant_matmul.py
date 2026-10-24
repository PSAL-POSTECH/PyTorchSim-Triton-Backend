"""W8A8 -- an int32 contraction, then a per-output-channel dequant epilogue."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 16, 32, 16            # tokens, hidden, output channels


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def int8_dequant_matmul_kernel(xq_ptr, wq_ptr, sx_ptr, sw_ptr, b_ptr,
                                   acc_ptr, o_ptr, M: tl.constexpr,
                                   K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)

        xq = tl.load(xq_ptr + rm[:, None] * K + rk[None, :])      # [M, K] int32

        # INTEGER CONTRACTION, one output channel at a time so every tile stays
        # rank 2 and the fold stays on the inner axis. Not a tl.dot: the array
        # has no integer MAC, so this has to reach the vector unit instead.
        acc = tl.zeros((M, N), dtype=tl.int32)
        for j in tl.static_range(N):
            wj = tl.load(wq_ptr + rk * N + j)                     # [K] int32
            aj = tl.sum(xq * wj[None, :], axis=1)                 # [M] int32
            acc = tl.where(rn[None, :] == j, aj[:, None], acc)

        off = rm[:, None] * N + rn[None, :]
        tl.store(acc_ptr + off, acc)

        # DEQUANT AS AN EPILOGUE. Neither scale could be here if either depended
        # on k; both do not, which is what per-token and per-channel mean.
        sx = tl.load(sx_ptr + rm)[:, None]
        sw = tl.load(sw_ptr + rn)[None, :]
        tl.store(o_ptr + off,
                 acc.to(tl.float32) * sx * sw + tl.load(b_ptr + rn)[None, :])

    return int8_dequant_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # int8-ranged values in int32 containers -- symmetric, so no zero point.
    return {"xq": torch.randint(-128, 128, (M * K,), generator=g,
                                dtype=torch.int32),
            "wq": torch.randint(-128, 128, (K * N,), generator=g,
                                dtype=torch.int32),
            "sx": torch.rand(M, generator=g, dtype=torch.float32) * 0.02 + 0.005,
            "sw": torch.rand(N, generator=g, dtype=torch.float32) * 0.02 + 0.005,
            "b": torch.randn(N, generator=g, dtype=torch.float32) * 0.1}


def reference(inputs):
    import torch
    # float64 for the integer matmul: every value here is exactly representable,
    # so this is the int32 result and not an approximation of it.
    xq = inputs["xq"].reshape(M, K).to(torch.float64)
    wq = inputs["wq"].reshape(K, N).to(torch.float64)
    acc = (xq @ wq).to(torch.int32)
    out = (acc.to(torch.float32) * inputs["sx"][:, None] * inputs["sw"][None, :]
           + inputs["b"][None, :])
    return {"acc": acc.reshape(-1).contiguous(),
            "o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="int8_dequant_matmul_kernel", kernel=kernel,
    signature={"xq_ptr": "*i32", "wq_ptr": "*i32", "sx_ptr": "*fp32",
               "sw_ptr": "*fp32", "b_ptr": "*fp32", "acc_ptr": "*i32",
               "o_ptr": "*fp32", "M": "constexpr", "K": "constexpr",
               "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("xq", IN, "int32", (M * K,)),
          Arg("wq", IN, "int32", (K * N,)),
          Arg("sx", IN, "float32", (M,)),
          Arg("sw", IN, "float32", (N,)),
          Arg("b", IN, "float32", (N,)),
          Arg("acc", OUT, "int32", (M * N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-passes/04-pytorchsim-materialize-layouts.mlir": {"pytorchsim.copy": 0}},
)
