"""`(i - N) % K` on a negative base: remsi takes the dividend's sign, so the values are in (-K, 0], not the
[0, K) a wrap on the axis walks. Breaks (wrong values) if visitOperandRemAsWrap folds an offset that may be
negative into a wrap instead of demoting the modulo."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K = 64, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_rem_negative_base_truncates_kernel(x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(o_ptr + i, tl.load(x_ptr + ((i - N) % K + K)))

    return ptr_rem_negative_base_truncates_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(2 * K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(N)
    return {"o": inputs["x"][torch.fmod(i - N, K) + K].clone()}


SPEC = KernelSpec(
    name="ptr_rem_negative_base_truncates_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "K": "constexpr"},
    constexprs={"N": N, "K": K},
    args=[Arg("x", IN, "float32", (2 * K,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
