"""A looped Welford reduction as Inductor writes it: `welford_reduce(..., roffset
== 0)` branches on the first trip, so three tiles come out of a tensor `scf.if`
-- upstream's kConditional, whose branch results take the conditional's layout.

convnextv2's layer norm once its reduction outgrows the block: the first trip of welford_reduce is a tensor scf.if. Left to the route below bufferize it was fused without layouts and refused.
"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, R = 64, 256, 64


def kernel():
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers

    @triton.jit
    def loop_welford_first_trip_if_kernel(x_ptr, mean_ptr, var_ptr,
                                          M: tl.constexpr, N: tl.constexpr,
                                          R: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, R)[None, :]
        mean = tl.zeros([M, R], tl.float32)
        m2 = tl.zeros([M, R], tl.float32)
        weight = tl.zeros([M, R], tl.float32)
        for off in range(0, N, R):
            x = tl.load(x_ptr + rows * N + (off + cols))
            mean, m2, weight = triton_helpers.welford_reduce(
                x, mean, m2, weight, off == 0)
        m, v, w = triton_helpers.welford(mean, m2, weight, 1)
        tl.store(mean_ptr + rows, m[:, None])
        tl.store(var_ptr + rows, (v / w)[:, None])

    return loop_welford_first_trip_if_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, N)
    return {"mean": x.mean(dim=1), "var": x.var(dim=1, unbiased=False)}


SPEC = KernelSpec(
    name="loop_welford_first_trip_if_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "mean_ptr": "*fp32", "var_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "R": "constexpr"},
    constexprs={"M": M, "N": N, "R": R},
    args=[Arg("x", IN, "float32", (M * N,)),
          Arg("mean", OUT, "float32", (M,)), Arg("var", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    # DECIDED ABOVE BUFFERIZE: the function is marked only where the layout pass
    # rewrote it -- an scf.if it could not rewrite sent the whole module below.
)
