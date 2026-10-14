"""A transcendental at the widest group this machine issues: 64 f32 is LMUL 8 at VLEN 256, and
the VCIX lowering must take it whole. It once refused a group wider than a fixed 2048 bits; this
breaks if that width, or legalize's cap, stops being VLEN x LMUL 8 read off the machine."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32768


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_math_at_lmul8_kernel(x_ptr, y_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        tl.store(y_ptr + r, tl.exp(tl.load(x_ptr + r)))

    return ops_math_at_lmul8_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"y": inputs["x"].exp()}


SPEC = KernelSpec(
    name="ops_math_at_lmul8_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE WIDTH IS THE CLAIM: narrower, and the refusal it pins cannot fire.
    # The claim is the WIDTH, not an SSA number: %n moves when allocation does.
    expect_ir={"08-transformed.mlir": {"math.exp": 1, ": vector<64xf32>": 1}},
)
