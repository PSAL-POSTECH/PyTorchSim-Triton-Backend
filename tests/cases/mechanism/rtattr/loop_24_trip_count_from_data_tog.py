"""Pins a loop bound and a mask resting on data as scalar runtime attributes: an empty fused_moe block runs
its K loop 0 times, so the TOG loads B 64 times over the grid, not 256; a loop it cannot count, or the
store's extent clamped up to 1, breaks it, and so does a TOG that runs the bound it was compiled with."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

E, K, N, BM, BN, BK = 8, 256, 256, 16, 64, 64
LIVE = [(1, 14), (5, 16), (5, 16), (5, 7)]
NB = 16
GRP = [g for g, _ in LIVE] + [E - 1] * (NB - len(LIVE))
ROWS = [r for _, r in LIVE] + [0] * (NB - len(LIVE))
M = NB * BM


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_24_trip_count_from_data_tog(a_ptr, b_ptr, grp_ptr, rows_ptr, c_ptr, K: tl.constexpr,
                                         N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                                         BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        e = tl.load(grp_ptr + pm)
        nr = tl.load(rows_ptr + pm)
        rm = tl.arange(0, BM)
        live = rm < nr
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, tl.where(nr > 0, K, 0), BK):
            a = tl.load(a_ptr + (pm * BM + rm)[:, None] * K + (k0 + rk)[None, :], mask=live[:, None], other=0.0)
            b = tl.load(b_ptr + e * (K * N) + (k0 + rk)[:, None] * N + rn[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + (pm * BM + rm)[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=live[:, None])

    return loop_24_trip_count_from_data_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": (torch.randn(M, K, generator=g) * 0.1).to(torch.bfloat16),
            "b": (torch.randn(E, K, N, generator=g) * 0.05).to(torch.bfloat16),
            "grp": torch.tensor(GRP, dtype=torch.int32), "rows": torch.tensor(ROWS, dtype=torch.int32),
            "c": torch.zeros(M, N, dtype=torch.bfloat16)}


def reference(inputs):
    import torch
    a, b, c = inputs["a"].float(), inputs["b"].float(), inputs["c"].clone()
    for i, (e, r) in enumerate(zip(GRP, ROWS)):
        c[i * BM:i * BM + r] = (a[i * BM:i * BM + r] @ b[e]).to(torch.bfloat16)
    return {"c": c}


SPEC = KernelSpec(
    name="loop_24_trip_count_from_data_tog", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "grp_ptr": "*i32", "rows_ptr": "*i32", "c_ptr": "*bf16",
               "K": "constexpr", "N": "constexpr", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"K": K, "N": N, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (E, K, N)),
          Arg("grp", IN, "int32", (NB,)), Arg("rows", IN, "int32", (NB,)), Arg("c", OUT, "bfloat16", (M, N))],
    expect_ir={"09-custom.mlir": {"torchsim.runtime_attribute": 2}},
    grid=(NB, N // BN), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
