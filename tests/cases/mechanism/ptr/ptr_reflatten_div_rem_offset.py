"""A row cut written as the fold writes it, from a range that does not start at 0.

The fold reads [R, n] as [R * k, n / k]: row p // k, chunk (p % k) * (n / k) plus depth a. With
p = arange(C * k, (C + R) * k) both halves of the cut carry the start, in elements: the access
must begin at element C * n whether or not the two halves are merged back into one axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, K, NC, C = 8, 4, 64, 3
W = NC // K


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_reflatten_div_rem_offset_kernel(x_ptr, o_ptr, R: tl.constexpr, K: tl.constexpr,
                                            NC: tl.constexpr, C: tl.constexpr):
        p = tl.arange(C * K, (C + R) * K)[:, None]
        a = tl.arange(0, NC // K)[None, :]
        v = tl.load(x_ptr + ((p // K) * NC + (a + (p % K) * (NC // K))))
        o = tl.arange(0, R * K)[:, None] * (NC // K) + a
        tl.store(o_ptr + o, v * 2.0)

    return ptr_reflatten_div_rem_offset_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C + R, NC, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"][C:].reshape(R * K, W) * 2.0}


SPEC = KernelSpec(
    name="ptr_reflatten_div_rem_offset_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "K": "constexpr",
               "NC": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "K": K, "NC": NC, "C": C},
    args=[Arg("x", IN, "float32", (C + R, NC)), Arg("o", OUT, "float32", (R * K, W))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.load": 1}},
)
