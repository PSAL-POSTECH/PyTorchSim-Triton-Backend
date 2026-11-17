"""A row's min and max of one bf16 tile, as torchao's activation quantization takes them (amin and amax of x):
two reduces of one input over the same dim are walked once, one loop carrying both accumulators, as XLA's loop
fusion takes them. Breaks if the two reduces go back to two loops, each reading and widening the row again.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_sibling_min_max_one_walk_kernel(x_ptr, mn_ptr, mx_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :]).to(tl.float32)
        tl.store(mn_ptr + rm, tl.min(x, axis=1))
        tl.store(mx_ptr + rm, tl.max(x, axis=1))

    return reduce_sibling_min_max_one_walk_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g).to(torch.bfloat16)}


def reference(inputs):
    x = inputs["x"].float()
    return {"mn": x.amin(dim=1), "mx": x.amax(dim=1)}


SPEC = KernelSpec(
    name="reduce_sibling_min_max_one_walk_kernel", kernel=kernel,
    signature={"x_ptr": "*bf16", "mn_ptr": "*fp32", "mx_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "bfloat16", (M, N)), Arg("mn", OUT, "float32", (M,)), Arg("mx", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
    # ONE WALK IS THE CLAIM: a single loop carries an accumulator (one iter_args, holding both).
    expect_ir={"08-transformed.mlir": {"iter_args": 1}},
)
