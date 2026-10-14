"""SwiGLU feed-forward block -- TWO matmuls whose results meet in one elementwise
chain.

torch equivalent: silu(x @ wg) * (x @ wu),  silu(t) = t * sigmoid(t)

WHY TWO MATMULS AND NOT ONE WITH AN EPILOGUE.  mixed/ already covers a matmul
followed by elementwise work. What is not covered is two matmuls that are
SIBLINGS: both read the same `x` tile, neither feeds the other, and their results
are combined afterwards. That makes three demands on one scratchpad at once --

  x        read by two accumulation loops, so its tile must outlive both
  gate     a matmul result consumed by a nonlinearity, not by a store
  up       a matmul result consumed as the OTHER operand of that nonlinearity

-- and the matrix unit pins the lane axis of a matmul's output. Two outputs
pinned independently then have to agree, because the multiply that joins them is
elementwise and elementwise has no relayout. A kernel with one matmul cannot
produce that constraint; this is the smallest one that can.

THE NONLINEARITY IS WRITTEN OUT, not called: `t / (1 + exp(-t))` rather than
`tl.sigmoid`, so what reaches the backend is arith + math.exp on a matmul result
rather than an extern call whose lowering is a separate question. The division is
deliberate too -- it is the one elementwise op here that is neither add nor
multiply, and it lands on a tile the array wrote.

SHAPES.  BM = BN = BK = 128 = the lane count, one lane tile of the matrix unit. K = 2 * BK so
the accumulation loop survives; both matmuls carry their own loop over it.

Complex-op tier: two sibling matmuls off one x tile, joined by a nonlinearity -- two independently pinned lane axes that must agree.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def swiglu_kernel(x_ptr, wg_ptr, wu_ptr, o_ptr,
                      N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        gate = tl.zeros((BM, BN), dtype=tl.float32)
        up = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(x_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            wg = tl.load(wg_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            wu = tl.load(wu_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            gate += tl.dot(a, wg)
            up += tl.dot(a, wu)

        silu = gate / (1.0 + tl.exp(-gate))
        tl.store(o_ptr + offs_m[:, None] * N + offs_n[None, :], silu * up)

    return swiglu_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(M * K, generator=g, dtype=torch.float32) * 0.1,
        "wg": torch.randn(K * N, generator=g, dtype=torch.float32) * 0.1,
        "wu": torch.randn(K * N, generator=g, dtype=torch.float32) * 0.1,
    }


def reference(inputs):
    import torch
    x = inputs["x"].reshape(M, K)
    gate = x @ inputs["wg"].reshape(K, N)
    up = x @ inputs["wu"].reshape(K, N)
    return {"o": (torch.nn.functional.silu(gate) * up).reshape(-1)}


SPEC = KernelSpec(
    name="swiglu_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "wg_ptr": "*fp32", "wu_ptr": "*fp32",
               "o_ptr": "*fp32", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("x", IN, "float32", (M * K,)),
        Arg("wg", IN, "float32", (K * N,)),
        Arg("wu", IN, "float32", (K * N,)),
        Arg("o", OUT, "float32", (M * N,)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
