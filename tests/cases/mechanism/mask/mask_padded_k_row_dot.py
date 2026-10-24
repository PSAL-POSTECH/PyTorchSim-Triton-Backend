"""GEMV as Inductor emits it, K = 768 in a block of 1024: the padding is the row sum's identity, so
pytorchsim-fit-to-numel cuts every buffer's in-lane axis to 768 and the sums match the padded kernel's."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, N, XB, RB = 768, 256, 128, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_padded_k_row_dot_kernel(x_ptr, w_ptr, y_ptr, K: tl.constexpr, N: tl.constexpr,
                                     XB: tl.constexpr, RB: tl.constexpr):
        cols = tl.program_id(0) * XB + tl.arange(0, XB)[:, None]
        r = tl.arange(0, RB)[None, :]
        rmask = r < K
        x = tl.load(x_ptr + r, rmask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols + N * r, rmask, other=0.0).to(tl.float32)
        acc = tl.where(rmask, x * w, 0.0)
        tl.store(y_ptr + cols, tl.sum(acc, 1)[:, None].to(tl.bfloat16))

    return mask_padded_k_row_dot_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(K, generator=g).to(torch.bfloat16),
            "w": torch.randn(K * N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    x = inputs["x"].float()
    w = inputs["w"].float().reshape(K, N)
    return {"y": (x @ w).to(inputs["x"].dtype)}


SPEC = KernelSpec(
    name="mask_padded_k_row_dot_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "w_ptr": "*bf16", "y_ptr": "*bf16",
               "K": "constexpr", "N": "constexpr", "XB": "constexpr", "RB": "constexpr"},
    constexprs={"K": K, "N": N, "XB": XB, "RB": RB},
    args=[Arg("x", IN, "bfloat16", (K,)), Arg("w", IN, "bfloat16", (K * N,)),
          Arg("y", OUT, "bfloat16", (N,))],
    grid=(N // XB,), reference=reference, make_inputs=make_inputs,
    rtol=2e-2, atol=2e-1,
)
