"""`(x // K) % H` off `pid*BLOCK + arange` gathers ROWS: the remainder is taken on the row index.

WHAT IT IS. Inductor's 1-D pointwise kernel adding a per-head bias to a [B, H, S, D] view of a
[B, S, H*D] buffer (CLIP's attention `addmm + mul + transpose + view`) reads the bias at
`x % D + D*((x // (S*D)) % H)`. The division is cut into rows (ptr_div_into_rows.py): the block
offset is runtime and `gcd(BLOCK, S*D)` rows of consecutive x share one quotient.

THE BUG. The `% H` then saw an unstructured, cut state and rebuildAsGatherScatter put the
per-element remainder, a [BLOCK] tensor, on the cut's row axis: a [BLOCK/g, ..] gather whose
index is [BLOCK], refused in a02 ("gather index shape [1024] is neither the tile ...").

NOW. A row cut's value is its row index alone, so `% H` is taken on the row index and the gather
stays one index per row. rebuildAsGatherScatter refuses any other cut state.

THE SHAPE: K = S*D = 640 is not a divisor of BLOCK = 1024 and g = 128 covers the lanes; N = 3840
is not a multiple of BLOCK, so the last program is masked; H = 3 wraps inside the block.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, H, S, D, BLOCK = 2, 3, 10, 64, 1024
N = B * H * S * D


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_div_into_rows_then_rem_kernel(bias_ptr, y_ptr, o_ptr, N: tl.constexpr,
                                          H: tl.constexpr, S: tl.constexpr,
                                          D: tl.constexpr, BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = x < N
        head = (x // (S * D)) % H
        b = tl.load(bias_ptr + (x % D + D * head), m)
        y = tl.load(y_ptr + x, m)
        tl.store(o_ptr + x, (b + y) * 0.125, m)

    return ptr_div_into_rows_then_rem_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"bias": torch.randn(H * D, generator=g, dtype=torch.float32),
            "y": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["y"].view(B, H, S, D)
    bias = inputs["bias"].view(1, H, 1, D)
    return {"o": ((x + bias) * 0.125).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_div_into_rows_then_rem_kernel", kernel=kernel,
    signature={"bias_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr", "H": "constexpr", "S": "constexpr",
               "D": "constexpr", "BLOCK": "constexpr"},
    constexprs={"N": N, "H": H, "S": S, "D": D, "BLOCK": BLOCK},
    args=[Arg("bias", IN, "float32", (H * D,)),
          Arg("y", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=((N + BLOCK - 1) // BLOCK,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
    # The bias read: one gather over BLOCK/g = 8 rows, folded back by one collapse_shape.
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 1,
                                       "tensor.collapse_shape": 1}},
)
