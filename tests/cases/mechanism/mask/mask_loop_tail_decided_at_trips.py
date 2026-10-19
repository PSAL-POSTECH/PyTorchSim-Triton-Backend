"""A sum over 3/2 blocks the way Inductor writes it: the loop masks `off + r < N` on every trip, and
only the last trip is short. The loop's bounds decide the mask on the full trip (true) and, inside
the last trip, on its vector loop's first half (true) and second half (false), so the kernel keeps
no compare and no select: each run of trips does its side. Breaks if the mask is built on every
trip again.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B = 16384
N = 3 * B // 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_loop_tail_decided_at_trips_kernel(x_ptr, o_ptr, N: tl.constexpr, B: tl.constexpr):
        r = tl.arange(0, B)[None, :]
        acc = tl.full((1, B), 0, tl.float32)
        for off in range(0, N, B):
            m = off + r < N
            x = tl.load(x_ptr + off + r, m, other=0.0)
            acc = tl.where(m, acc + x, acc)
        tl.store(o_ptr + tl.arange(0, 1)[:, None], tl.sum(acc, axis=1)[:, None])

    return mask_loop_tail_decided_at_trips_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g)}


def reference(inputs):
    return {"o": inputs["x"].sum().reshape(1, 1)}


SPEC = KernelSpec(
    name="mask_loop_tail_decided_at_trips_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "B": "constexpr"},
    constexprs={"N": N, "B": B},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (1, 1))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-3,
    # THE CUT IS THE CLAIM: no trip builds the mask, and no loop says it kept one.
    expect_ir={"08-transformed.mlir": {"arith.cmpi": 0, "arith.select": 0,
                                       "torchsim.mask_kept": 0}},
)
