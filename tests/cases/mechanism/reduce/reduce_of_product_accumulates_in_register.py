"""`acc += tl.sum(x[:, None] * b, axis=0)`, a GEMV row: the product is fused into the reduce and summed
in a register, never stored as a K x N tile. Breaks if torchsim-fuse stops restating the reduce as a
generic (05-fuse then holds 2 named reduces) or vectorize stops taking a reduction generic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, N = 64, 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_of_product_kernel(a_ptr, b_ptr, c_ptr):
        rk = tl.arange(0, 32)
        rn = tl.arange(0, 256)
        x = tl.load(a_ptr + rk).to(tl.float32)
        b = tl.load(b_ptr + rk[:, None] * 256 + rn[None, :]).to(tl.float32)
        acc = tl.sum(x[:, None] * b, axis=0)
        x = tl.load(a_ptr + 32 + rk).to(tl.float32)
        b = tl.load(b_ptr + (32 + rk)[:, None] * 256 + rn[None, :]).to(tl.float32)
        acc += tl.sum(x[:, None] * b, axis=0)
        tl.store(c_ptr + rn, acc.to(tl.bfloat16))

    return reduce_of_product_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(1, K, generator=g, dtype=torch.float32).to(torch.bfloat16),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.bfloat16),
    }


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"].to(torch.float32),
                              inputs["b"].to(torch.float32)).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="reduce_of_product_kernel",
    kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16"},
    constexprs={},
    args=[Arg("a", IN, "bfloat16", (1, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "bfloat16", (1, N))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
    expect_ir={"08-passes/05-torchsim-fuse.mlir": {"linalg.reduce": 0, "linalg.generic": 3}},
)
