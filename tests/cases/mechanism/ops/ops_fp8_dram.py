"""fp8 in DRAM: an *fp8e5 argument loaded and widened at once, so 8 bits cross the bus and the
storage retype (fp8 -> i8 in every type) is all that carries it. Breaks if the retype misses a
memref, if the runtime writes the input as a float cast instead of its bits, or extf loses the format."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_fp8_dram_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r).to(tl.float32)   # 8 bits off the bus, widen at once
        tl.store(o_ptr + r, x * x)

    return ops_fp8_dram_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)
                      .to(torch.float8_e5m2)}


def reference(inputs):
    f = inputs["x"].float()
    return {"o": f * f}


SPEC = KernelSpec(
    name="ops_fp8_dram_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp8e5", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float8_e5m2", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="pass",
)
