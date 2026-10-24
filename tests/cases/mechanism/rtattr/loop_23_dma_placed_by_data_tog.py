"""Pins a DMA address resting on data as a scalar runtime attribute: fused_moe's grouped GEMM reads its
expert id and loads that expert's B block. The TOG reads the id with togsim::scalar and places B at 65536
for key 0 and 327680 for key 1; reading it as 0, or failing the TOG's build on the spad read, breaks it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

E, K, N, BM, BN, BK = 8, 256, 256, 16, 64, 64
SIZES = [0, 14, 0, 0, 0, 49, 0, 1]
BLOCK_EXPERT = [e for e, s in enumerate(SIZES) for _ in range(-(-s // BM))]
M = len(BLOCK_EXPERT) * BM


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_23_dma_placed_by_data_tog(a_ptr, b_ptr, ids_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr,
                                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        e = tl.load(ids_ptr + pm)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :])
            b = tl.load(b_ptr + e * (K * N) + (k0 + rk)[:, None] * N + rn[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))

    return loop_23_dma_placed_by_data_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": (torch.randn(M, K, generator=g) * 0.1).to(torch.bfloat16),
            "b": (torch.randn(E, K, N, generator=g) * 0.05).to(torch.bfloat16),
            "ids": torch.tensor(BLOCK_EXPERT, dtype=torch.int32)}


def reference(inputs):
    import torch
    a, b = inputs["a"].float(), inputs["b"].float()
    return {"c": torch.cat([a[i * BM:(i + 1) * BM] @ b[e] for i, e in enumerate(BLOCK_EXPERT)]).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="loop_23_dma_placed_by_data_tog", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "ids_ptr": "*i32", "c_ptr": "*bf16",
               "K": "constexpr", "N": "constexpr", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"K": K, "N": N, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (E, K, N)),
          Arg("ids", IN, "int32", (len(BLOCK_EXPERT),)), Arg("c", OUT, "bfloat16", (M, N))],
    expect_ir={"09-custom.mlir": {"pytorchsim.runtime_attribute": 1}},
    grid=(len(BLOCK_EXPERT), N // BN), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
