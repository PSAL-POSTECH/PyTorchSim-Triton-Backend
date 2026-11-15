"""A row max and min over 1024 f32 columns: legalize-vectors cuts the reduce loop to LMUL 4 and, as two sets of
LMUL-4 partial vectors span the widest register group and the register file holds them, carries two sets, each
folding every other chunk, combined after the loop; the in-order core overlaps one chunk's load -> combine chain
with the other's. Breaks if the loop carries one set again: one vector load in the loop instead of two."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reduce_two_accumulator_sets_kernel(x_ptr, lo_ptr, hi_ptr, M: tl.constexpr, N: tl.constexpr):
        rows = tl.arange(0, M)[:, None]
        x = tl.load(x_ptr + rows * N + tl.arange(0, N)[None, :])
        tl.store(lo_ptr + rows, tl.min(x, 1)[:, None])
        tl.store(hi_ptr + rows, tl.max(x, 1)[:, None])

    return ops_reduce_two_accumulator_sets_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, N)
    return {"lo": x.amin(dim=1), "hi": x.amax(dim=1)}


SPEC = KernelSpec(
    name="ops_reduce_two_accumulator_sets_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "lo_ptr": "*fp32", "hi_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)), Arg("lo", OUT, "float32", (M,)), Arg("hi", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
    expect_ir={"08-transformed.mlir": {"affine.vector_load": 2}},
)
