"""One fp32 exp of a 128x128 tile read by a row sum and by a dot (attention's p): the dot wants p
in one encoding and the sum in another, and the two encodings are the same linear layout -- every
lane holds the same elements in both -- so the convert between them moves nothing and costs 0.
The layout pass then keeps one exp and converts, instead of computing exp a second time in the
dot's encoding. Pins one math.exp in 04-anchored and in 05-layouts. torch: (exp(s) @ v) / rowsum."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, D = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_exp_feeds_rowsum_and_dot(s_ptr, v_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr,
                                      D: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rd = tl.arange(0, D)
        s = tl.load(s_ptr + rm[:, None] * N + rn[None, :])
        v = tl.load(v_ptr + rn[:, None] * D + rd[None, :])
        p = tl.exp(s)
        l = tl.sum(p, axis=1)
        o = tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        tl.store(o_ptr + rm[:, None] * D + rd[None, :], o / l[:, None])

    return tile_exp_feeds_rowsum_and_dot


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"s": torch.randn(M, N, generator=g, dtype=torch.float32),
            "v": torch.randn(N, D, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    p = torch.exp(inputs["s"])
    return {"o": (p.to(torch.bfloat16).float() @ inputs["v"].float()) / p.sum(1, keepdim=True)}


SPEC = KernelSpec(
    name="tile_exp_feeds_rowsum_and_dot", kernel=kernel,
    signature={"s_ptr": "*fp32", "v_ptr": "*bf16", "o_ptr": "*fp32", "M": "constexpr",
               "N": "constexpr", "D": "constexpr"},
    constexprs={"M": M, "N": N, "D": D},
    args=[Arg("s", IN, "float32", (M, N)), Arg("v", IN, "bfloat16", (N, D)),
          Arg("o", OUT, "float32", (M, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect_ir={"04-anchored.mlir": {"math.exp": 1},
               "05-layouts.mlir": {"math.exp": 1}},
)
