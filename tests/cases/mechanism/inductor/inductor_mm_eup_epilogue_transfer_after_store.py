"""Inductor's resident mm template with an exp epilogue: the epilogue's EUP loop stores each trip's output into a
staging buffer and transfers it out, and the transfer used to run before the store.

WHAT IT PINS. schedule_vpu_stream pipelines an EUP loop (the pushes DEPTH groups ahead of their pops) and then
orders the body by phase, keeping the stores in their order. The transfer out of the staging buffer is no store
and shares no value with them, only the buffer, so it was ordered with the loads, ahead of the store it sends.

    before   max_abs_err 963.0 -- columns 256..767 of every row are 0: the trips the pipelined loop ran sent
             their staging buffer before writing it
    after    max_abs_err 0.14 (bf16 rounding of the product)

An op that touches a buffer another op of the trip wrote before it now stays after that op
(tests/passes/eup_pipeline_cases.py a_copy_out_of_what_a_trip_stored_stays_after_the_store pins the order itself).

FOUND BY Qwen3-8B's bf16 MLP at 16 tokens, silu(gate) * up: relative error 1 from the up mm whose epilogue
reads gate through exp; at 1 and 8 tokens the template took another shape. This is that kernel, the gate input
in place of the silu, at M 16, N 8192, K 512.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 16, 8192, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def inductor_mm_eup_epilogue_transfer_after_store(arg_A, arg_B, in_ptr2, out_ptr1):
        TILES: tl.constexpr = 1
        BLOCK_M: tl.constexpr = 128
        BLOCK_N: tl.constexpr = 8192
        BLOCK_K: tl.constexpr = 512
        M = 16
        N = 8192
        K = 512
        rk = tl.arange(0, BLOCK_K)
        rn = tl.arange(0, BLOCK_N)
        b = tl.load(arg_B + rk[:, None] * 1 + rn[None, :] * 512,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        for t in range(0, tl.cdiv(M, TILES * BLOCK_M)):
            rm = (TILES * t + 0) * BLOCK_M + tl.arange(0, BLOCK_M)
            a = tl.load(arg_A + rm[:, None] * 512 + rk[None, :] * 1,
                        mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
            acc = tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
            out_m = rm[:, None]
            out_n = rn[None, :]
            mask = (out_m < M) & (out_n < N)
            xindex = out_n + 8192 * out_m
            tmp1 = acc.to(tl.bfloat16).to(tl.float32)
            tmp3 = tl.load(in_ptr2 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), mask).to(tl.float32)
            tmp5 = tl.exp(-tmp3)
            tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp5 * tmp1, mask)

    return inductor_mm_eup_epilogue_transfer_after_store


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M * K, generator=g).to(torch.bfloat16),
            "b": torch.randn(N * K, generator=g).to(torch.bfloat16),
            "c": torch.randn(M * N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    a = inputs["a"].float().reshape(M, K)
    b = inputs["b"].float().reshape(N, K)
    c = inputs["c"].float().reshape(M, N)
    return {"out": (torch.exp(-c) * (a @ b.t()).to(torch.bfloat16).float()).reshape(-1)}


SPEC = KernelSpec(
    name="inductor_mm_eup_epilogue_transfer_after_store",
    kernel=kernel,
    signature={"arg_A": "*bf16", "arg_B": "*bf16", "in_ptr2": "*bf16", "out_ptr1": "*fp32"},
    constexprs={},
    args=[
        Arg("a", IN, "bfloat16", (M * K,)),
        Arg("b", IN, "bfloat16", (N * K,)),
        Arg("c", IN, "bfloat16", (M * N,)),
        Arg("out", OUT, "float32", (M * N,)),
    ],
    grid=(1,),
    graph_prefetch=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=0.5,
)
