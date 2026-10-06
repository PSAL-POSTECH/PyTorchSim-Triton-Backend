"""An index Inductor unflattened and this backend puts back: `(x%K) + K*(x//K)`.

WHAT IT PINS: ttclean's _reflatten_div_rem. The identity is `x = K*(x//K) + (x%K)`, exact for
every x under truncating division, so the pair is a rename rather than two
axes. Left standing, the `//` ends triton-shared's structured analysis at the
first sight of it and the access becomes a gather over an index it did not need.

WHAT PROVES IT, with _reflatten_div_rem registered and without it:
    on   tts.make_tptr + tts.load
    off  tts.make_tptr + tts.gather

_reflatten_div_rem: the stride relation b == K*a is the discriminator -- RoPE's interleave has the same shape and deliberately fails it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, N = 8, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_reflatten_div_rem_kernel(x_ptr, o_ptr, K: tl.constexpr,
                                     N: tl.constexpr):
        i = tl.arange(0, N)
        col = i % K
        row = i // K
        v = tl.load(x_ptr + (col + K * row))     # == x_ptr + i
        tl.store(o_ptr + i, v)

    return ptr_reflatten_div_rem_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].clone()}


SPEC = KernelSpec(
    name="ptr_reflatten_div_rem_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "K": "constexpr",
               "N": "constexpr"},
    constexprs={"K": K, "N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"03-ttshared.mlir": {"tts.gather": 0, "tts.load": 1}},
)
