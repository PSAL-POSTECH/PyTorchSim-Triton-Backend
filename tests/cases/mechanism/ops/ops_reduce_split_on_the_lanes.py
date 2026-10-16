"""A softmax over a dot's rows, 4 rows by 1024 columns: the row reduces split on the lanes first.

WHAT IT PINS. The array takes a dot's product with N on the lanes, and a reduce over the last
axis wants that axis off them, rows on the lanes. 4 rows do not fill the lanes, so the dot stays
as it is; left there, the whole [4, 1024] score crosses the lanes for the row max and sum, and
p = exp(s - m) is computed once in each layout. Here the reduced axis is on the lanes and holds
1024 = 8 x 128 elements, so anchor-layouts reduces it in two steps: each lane first reduces the 8
elements it holds, in the dot's layout, and only the [4, 128] partial crosses. p stays in the
dot's layout and is computed once.

    before   1 crossing of [4, 1024] (8 rounds), 2 gathers, math.exp = 2, tiles [6, 2642] cycles
    after    2 crossings of [4, 128] (1 round each), 2 gathers, math.exp = 1, tiles [6, 1916] cycles,
             the same max_abs_err

torch equivalent: softmax(q @ kt) @ v
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, D, N = 4, 128, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reduce_split_on_the_lanes_kernel(
            q_ptr, kt_ptr, v_ptr, o_ptr, R: tl.constexpr, D: tl.constexpr, N: tl.constexpr):
        r = tl.arange(0, R)
        d = tl.arange(0, D)
        n = tl.arange(0, N)
        q = tl.load(q_ptr + r[:, None] * D + d[None, :])
        kt = tl.load(kt_ptr + d[:, None] * N + n[None, :])
        v = tl.load(v_ptr + n[:, None] * D + d[None, :])
        s = tl.dot(q, kt, out_dtype=tl.float32)
        p = tl.exp(s - tl.max(s, axis=1)[:, None])
        l = tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        tl.store(o_ptr + r[:, None] * D + d[None, :], acc / l[:, None])

    return ops_reduce_split_on_the_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"q": torch.randn(R * D, generator=g).to(torch.bfloat16),
            "kt": torch.randn(D * N, generator=g).to(torch.bfloat16),
            "v": torch.randn(N * D, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    s = inputs["q"].float().reshape(R, D) @ inputs["kt"].float().reshape(D, N)
    return {"o": torch.softmax(s, -1) @ inputs["v"].float().reshape(N, D)}


SPEC = KernelSpec(
    name="ops_reduce_split_on_the_lanes_kernel", kernel=kernel,
    signature={"q_ptr": "*bf16", "kt_ptr": "*bf16", "v_ptr": "*bf16", "o_ptr": "*fp32",
               "R": "constexpr", "D": "constexpr", "N": "constexpr"},
    constexprs={"R": R, "D": D, "N": N},
    args=[Arg("q", IN, "bfloat16", (R * D,)), Arg("kt", IN, "bfloat16", (D * N,)),
          Arg("v", IN, "bfloat16", (N * D,)), Arg("o", OUT, "float32", (R, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    # THE SPLIT IS THE CLAIM -- p is computed once, in the dot's layout.
    expect_ir={"08-transformed.mlir": {"math.exp": 1}},
)
