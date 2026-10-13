"""Regression guard (the tree before the fusion passes it too): two dots per window cast to bf16
through one staging buffer, each cast at its pops after the MVOUT that drained it before. Breaks
if a cast reads an unfinished sum or lands on bytes still being stored (values)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 1024, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_gemm_epilogue_shared_staging(a_ptr, b_ptr, c_ptr):
        ok = tl.arange(0, 512)
        om = tl.arange(0, 256)
        a0 = tl.load(a_ptr + om[:, None] * 512 + ok[None, :])
        a1 = tl.load(a_ptr + (256 + om)[:, None] * 512 + ok[None, :])
        for n in range(0, 1024, 256):
            on = n + tl.arange(0, 256)
            b = tl.load(b_ptr + ok[:, None] * 1024 + on[None, :])
            acc = tl.dot(a0, b, out_dtype=tl.float32)
            tl.store(c_ptr + om[:, None] * 1024 + on[None, :], acc.to(tl.bfloat16))
            acc = tl.dot(a1, b, out_dtype=tl.float32)
            tl.store(c_ptr + (256 + om)[:, None] * 1024 + on[None, :], acc.to(tl.bfloat16))

    return tile_gemm_epilogue_shared_staging


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"c": (inputs["a"].float() @ inputs["b"].float()).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="tile_gemm_epilogue_shared_staging", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16"},
    constexprs={},
    args=[Arg("a", IN, "bfloat16", (M, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "bfloat16", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
)
