"""GRID: a WRAPPED axis refined to meet a split, carrying its offset.

`(C + p)//K` and `(C + p)%K` want one axis cut at K, and the `%` side wraps. A
wrap survives the cut only when it equals the new outer stride -- the outer term
is then whole laps and vanishes -- and the OFFSET goes with it, because it sits
inside the modulo.

The wrap and the offset both ride the inner half of the cut: sizes [128, 2], strides [1, 512], offsets [0, 1024], shape [0, 1024], base_offset 1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, K, HALF, C = 256, 2, 512, 2
NX = (C + N - 1) // K + HALF * ((C + N - 1) % K) + 1        # 641


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_offset_survives_the_cut_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                            K: tl.constexpr, HALF: tl.constexpr,
                                            C: tl.constexpr):
        p = tl.arange(0, N)
        q = C + p
        v = tl.load(x_ptr + ((q // K) + HALF * (q % K)))
        tl.store(o_ptr + p, v)

    return wrap_offset_survives_the_cut_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    q = C + torch.arange(N)
    return {"o": inputs["x"][(q // K) + HALF * (q % K)]}


SPEC = KernelSpec(
    name="wrap_offset_survives_the_cut_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K": "constexpr", "HALF": "constexpr", "C": "constexpr"},
    constexprs={"N": N, "K": K, "HALF": HALF, "C": C},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE CLAIM IS THE RUNG -- a gather computes the same addresses one at a
    # time. Refusing a non-zero offset on the wrapped axis, which is what this
    # used to do, costs TWO rungs: the refusal lands after the division has
    # already succeeded, so nothing rebuilds it as an index table.
    expect_ir={"03-linalg.mlir": {"tts.gather": 0, "tts.make_tptr": 2}},
)
