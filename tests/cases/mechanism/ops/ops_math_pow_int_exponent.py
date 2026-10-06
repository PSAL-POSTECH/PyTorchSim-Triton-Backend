"""libdevice.pow(x, int32) arrives as __nv_powif, which triton-shared maps to math.fpowi and
upstream's math-expand-ops opens: a sitofp, then math.powf on a negative base. Breaks if the
table loses the symbol or fpowi leaves the upstream list (refused), or powf goes upstream (NaN).

convert_unsupported_ops, POSITIVE. (fp32, int32) selects __nv_powif, which triton-shared's extern table maps to math.fpowi. x straddles zero and n covers [-3, 3], so odd and even exponents both meet a negative base and n = 0 gives 1. The tolerance is loose because exp(n*log|x|) is not the exact integer power.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_pow_int_exponent_kernel(x_ptr, n_ptr, o_ptr,
                                         BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r)
        n = tl.load(n_ptr + r)
        tl.store(o_ptr + r, libdevice.pow(x, n))

    return ops_math_pow_int_exponent_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    mag = torch.rand(N, generator=g, dtype=torch.float32) * 2.0 + 0.5
    sign = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0)
    n = torch.randint(-3, 4, (N,), generator=g, dtype=torch.int32)
    return {"x": (mag * sign).contiguous(), "n": n.contiguous()}


def reference(inputs):
    import torch
    x, n = inputs["x"], inputs["n"]
    return {"o": torch.pow(x.to(torch.float64),
                           n.to(torch.float64)).to(torch.float32).contiguous()}


SPEC = KernelSpec(
    name="ops_math_pow_int_exponent_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "n_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("n", IN, "int32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-5,
)
