"""f16 arithmetic in a VECTOR, which needs Zvfh and not Zfh.

The suite's other fp16 kernels are gemms: they load f16, bitcast, and accumulate
in f32, so nothing ever asks for `<N x half>` arithmetic.  This one adds in f16.

WHAT IT PINS: `-mattr=+zvfh` in pipeline.py.  `zfh` is the SCALAR half extension
and says nothing about vector element types, so with it llc cannot legalise
`arith.addf : vector<8xf16>` and scalarises the whole loop --

    with +zfh    flh / fadd.h / fsh, eight times, no vector instruction
    with +zvfh   vle16.v / vfadd.vv / vse16.v

-- and the scalar form is not merely slow.  A scalar load addresses the
scratchpad FLAT while the DMA laid it out per lane, so only the element at lane
0 offset 0 lands where it was written: measured 63 of 64 elements wrong, the
output holding the INPUT unchanged for all of them.

Spike runs the vector form on `--isa rv64gcv_zfh`, which rejects the NAME zvfh
("unsupported extension") while executing vfadd.vv at e16 perfectly well.

f16 as an ARITHMETIC type, not a storage one. The gemms load f16 and compute in f32; this is the only kernel that asks llc for <N x half> arithmetic, which is what Zvfh names and Zfh does not.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_f16_elementwise_kernel(x_ptr, o_ptr, N: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(x_ptr + i)
        tl.store(o_ptr + i, v + v)

    return ops_f16_elementwise_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32).half()}


def reference(inputs):
    x = inputs["x"]
    return {"o": x + x}


SPEC = KernelSpec(
    name="ops_f16_elementwise_kernel", kernel=kernel,
    signature={"x_ptr": "*fp16", "o_ptr": "*fp16", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float16", (N,)), Arg("o", OUT, "float16", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
