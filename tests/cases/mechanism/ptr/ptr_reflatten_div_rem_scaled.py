"""`(x%K)*S + (x//K)*(K*S)` is `x*S`: the reflattened index carries a stride S != 1.

The scale is a splat of the index's tensor type; built from the tensor type itself, TritonToStructured crashed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, S, N = 8, 4, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_reflatten_div_rem_scaled_kernel(x_ptr, o_ptr, K: tl.constexpr,
                                            S: tl.constexpr, N: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(x_ptr + ((i % K) * S + (i // K) * (K * S)))     # == x_ptr + i*S
        tl.store(o_ptr + i, v * 2.0)

    return ptr_reflatten_div_rem_scaled_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N * S, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"][::S] * 2.0}


SPEC = KernelSpec(
    name="ptr_reflatten_div_rem_scaled_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "K": "constexpr",
               "S": "constexpr", "N": "constexpr"},
    constexprs={"K": K, "S": S, "N": N},
    args=[Arg("x", IN, "float32", (N * S,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.load": 1}},
)
