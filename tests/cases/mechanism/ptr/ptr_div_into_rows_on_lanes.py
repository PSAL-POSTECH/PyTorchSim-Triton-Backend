"""The row cut of ptr_div_into_rows.py is NOT taken when the rows would sit on lanes.

Here g = gcd(BLOCK, K) = gcd(256, 96) = 32 is shorter than the 128 lanes, so
the [8, 32] tile spreads its rows over lanes and one index per row would have
to differ from lane to lane -- the per-element gather again, which the layout
pass then refuses on its mask. `divideIntoRows` reads that off the tile's
linear layout and leaves the division as it was: one 256-wide gather per load,
no collapse, and the kernel still runs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, K, BLOCK = 16, 96, 256
N = T * K


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_div_into_rows_on_lanes_kernel(gu_ptr, o_ptr, N: tl.constexpr,
                                 K: tl.constexpr, BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = x < N
        col = x % K
        row = x // K
        a = tl.load(gu_ptr + (col + 2 * K * row), m)
        b = tl.load(gu_ptr + (K + col + 2 * K * row), m)
        tl.store(o_ptr + x, a / (1.0 + tl.exp(-a)) * b, m)

    return ptr_div_into_rows_on_lanes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"gu": torch.randn(T * 2 * K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    gu = inputs["gu"].view(T, 2 * K)
    return {"o": (torch.nn.functional.silu(gu[:, :K]) * gu[:, K:]).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_div_into_rows_on_lanes_kernel", kernel=kernel,
    signature={"gu_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr",
               "K": "constexpr", "BLOCK": "constexpr"},
    constexprs={"N": N, "K": K, "BLOCK": BLOCK},
    args=[Arg("gu", IN, "float32", (T * 2 * K,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(N // BLOCK,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
    # The division keeps its per-element gather: no row cut, no collapse.
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 2,
                                       "tensor.collapse_shape": 0}},
)
