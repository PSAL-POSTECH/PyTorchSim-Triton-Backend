"""GEMV y[N] = sum_k x[k]*B[k,N] with N=512 on 128 lanes: each lane holds 4 of N per K row, so the
reduce over K walks vectors of 4 one row at a time. tritonpytorchsim-pack-reduction-rows takes d
contiguous rows per trip and folds them in-lane once; breaks if the pass stops firing (no slices)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, N = 256, 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_rows_packed_into_the_vector_kernel(a_ptr, b_ptr, c_ptr):
        rk = tl.arange(0, 256)
        rn = tl.arange(0, 512)
        x = tl.load(a_ptr + rk).to(tl.float32)
        b = tl.load(b_ptr + rk[:, None] * 512 + rn[None, :]).to(tl.float32)
        acc = tl.sum(x[:, None] * b, axis=0)
        tl.store(c_ptr + rn, acc.to(tl.bfloat16))

    return reduce_rows_packed_into_the_vector_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(1, K, generator=g, dtype=torch.float32).to(torch.bfloat16),
            "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.bfloat16)}


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"].to(torch.float32),
                              inputs["b"].to(torch.float32)).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="reduce_rows_packed_into_the_vector_kernel", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16"},
    constexprs={},
    args=[Arg("a", IN, "bfloat16", (1, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "bfloat16", (1, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
    expect_ir={"09-custom.mlir": {"vector.extract_strided_slice": 8}},
)
