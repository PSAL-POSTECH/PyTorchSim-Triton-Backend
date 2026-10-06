"""A transcendental at the widest group this machine issues: 128 f32 is LMUL 8 at
VLEN 512, and the VCIX lowering must accept it -- it once refused anything past
2048 bits, LMUL 8 at VLEN 256, a width legalize never caps to.

exp over 128 elements a lane: one LMUL-8 group at VLEN 512, which the VCIX lowering has to take whole.
"""
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
    expect_ir={"04-transformed.mlir": {"math.exp": 1, ": vector<128xf32>": 1}},
)
