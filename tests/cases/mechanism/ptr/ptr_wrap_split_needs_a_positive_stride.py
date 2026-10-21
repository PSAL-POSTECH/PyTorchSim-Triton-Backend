"""`(r - C/4) % W + (r // K) * S` on one axis: a wrap whose offset is negative, cut at K by the
division. W divides the offset, so the wrap-split branch of refineStateAxis took it as whole laps
from a lap boundary and dropped the offset -- but `srem` keeps the dividend's sign, so the rows
below zero read `(r - 32) srem 4` in {-3..0}, not the {0..3} the split states. The same branch
would mint negative axis sizes for a stride below 0 (`-8 % 4 == 0` and `4 / -1 == -4` pass its
divisibility tests); that stride shape reaches it from no kernel found yet (it becomes a gather).
Breaks if the wrap split loses its `s > 0` and non-negative offset guards (Spike checks values).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, C, W, K, S = 128, 128, 4, 8, 100
SIZE = (N // K - 1) * S + 2 * W


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_split_needs_a_positive_stride_kernel(x_ptr, o_ptr, N: tl.constexpr,
                                                      C: tl.constexpr, W: tl.constexpr,
                                                      K: tl.constexpr, S: tl.constexpr):
        r = tl.arange(0, N)
        v = tl.load(x_ptr + W + ((r - C // 4) % W + (r // K) * S))
        tl.store(o_ptr + r, v)

    return ptr_wrap_split_needs_a_positive_stride_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(SIZE, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(N)
    return {"o": inputs["x"][W + torch.fmod(r - C // 4, W) + (r // K) * S].clone()}


SPEC = KernelSpec(
    name="ptr_wrap_split_needs_a_positive_stride_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr", "C": "constexpr",
               "W": "constexpr", "K": "constexpr", "S": "constexpr"},
    constexprs={"N": N, "C": C, "W": W, "K": K, "S": S},
    args=[Arg("x", IN, "float32", (SIZE,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
