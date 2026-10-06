"""Flash attention -- one kernel that is every hard thing in the suite at once.

torch equivalent: softmax(q @ k.T) @ v

WHY THIS IS THE TOP OF THE TIER.  Every other kernel here isolates one shape.
This one has all of them in a single `scf.for`, and they interact:

  two matmuls with different operands   q@kT feeds the softmax that weights p@v
  a REDUCTION INSIDE A LOOP             tl.max / tl.sum per KV block
  loop-carried reduction state          m_i, l_i and acc are iter_args, so the
                                        running max and running sum cross the
                                        back edge as tensors
  a rescale of the accumulator          acc = acc * alpha[:, None] + ...,
                                        i.e. a per-row broadcast applied to a
                                        MATMUL RESULT, inside the loop
  three pointer families advancing      q is loop-invariant, k and v step by BN

The online-softmax rescale is the part with no analogue anywhere else: `alpha`
is one value per row, computed from two loop-carried rank-1 tensors, and it
multiplies both a rank-1 tensor (l_i) and a rank-2 tile (acc) in the same
iteration. So the lane axis chosen for the rank-1 state and the one chosen for
the matmul's output tile have to AGREE, and they are decided by different rules --
the rank-1 fold has no axis left over (see reduce/), the matmul's is pinned by
the systolic array. A kernel that gets one of them wrong is silently wrong, not
refused, because both are legal layouts on their own.

K IS LOADED TRANSPOSED rather than through `tl.trans`: the pointer arithmetic
`k_ptr + offs_d[:, None] + n[None, :] * D` produces the (D, BN) tile `tl.dot`
wants directly, which keeps the address structured (PtrAnalysis stays on its
affine path) instead of asking a later pass to relayout a loaded tile.

SHAPES.  BM = D = BN = 128 = the lane count and the systolic edge, so no tile is
split and every matmul fills the array. N_CTX = 2 * BN on purpose: one KV block
would let triton unroll the loop away and take the loop-carried state with it,
which is the whole point of the kernel.

Complex-op tier, top rung: loop-carried softmax state, two matmuls, a per-row rescale of a matmul result, all inside one scf.for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M_CTX = 256      # queries
N_CTX = 256      # keys/values -- 2 KV blocks
D = 128          # head dim, in-lane and the systolic edge
BM = 128         # queries per program == TORCHSIM_COMPILE_VECTORLANE_SIZE
BN = 128         # keys per iteration


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                         N_CTX: tl.constexpr, D: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr,
                         SCALE: tl.constexpr):
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, BN)

        q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :])

        m_i = tl.full((BM,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)
        acc = tl.zeros((BM, D), dtype=tl.float32)

        for start_n in range(0, N_CTX, BN):
            n = start_n + offs_n
            # (D, BN): k read transposed by its own addressing, not by tl.trans.
            kt = tl.load(k_ptr + n[None, :] * D + offs_d[:, None])
            v = tl.load(v_ptr + n[:, None] * D + offs_d[None, :])

            qk = tl.dot(q, kt) * SCALE
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v)
            m_i = m_new

        tl.store(o_ptr + offs_m[:, None] * D + offs_d[None, :],
                 acc / l_i[:, None])

    return attention_kernel


SCALE = 1.0 / (D ** 0.5)


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn(M_CTX * D, generator=g, dtype=torch.float32) * 0.5,
        "k": torch.randn(N_CTX * D, generator=g, dtype=torch.float32) * 0.5,
        "v": torch.randn(N_CTX * D, generator=g, dtype=torch.float32) * 0.5,
    }


def reference(inputs):
    import torch
    q = inputs["q"].reshape(M_CTX, D)
    k = inputs["k"].reshape(N_CTX, D)
    v = inputs["v"].reshape(N_CTX, D)
    p = torch.softmax(q @ k.T * SCALE, dim=1)
    return {"o": (p @ v).reshape(-1)}


SPEC = KernelSpec(
    name="attention_kernel",
    kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "o_ptr": "*fp32", "N_CTX": "constexpr", "D": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "SCALE": "constexpr"},
    constexprs={"N_CTX": N_CTX, "D": D, "BM": BM, "BN": BN, "SCALE": SCALE},
    args=[
        Arg("q", IN, "float32", (M_CTX * D,)),
        Arg("k", IN, "float32", (N_CTX * D,)),
        Arg("v", IN, "float32", (N_CTX * D,)),
        Arg("o", OUT, "float32", (M_CTX * D,)),
    ],
    grid=(M_CTX // BM,),
    reference=reference,
    make_inputs=make_inputs,
    # Two chained matmuls and a rescaled accumulation, none in torch's order.
    rtol=1e-3, atol=1e-3,
)
