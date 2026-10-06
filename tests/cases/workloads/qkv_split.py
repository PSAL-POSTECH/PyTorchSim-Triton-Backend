"""GPT-2's fused QKV projection -- one weight matrix, three column ranges.

    qkv = x @ W_qkv          W_qkv is [D, 3D], so qkv is [T, 3D]
    q, k, v = qkv[:, :D], qkv[:, D:2D], qkv[:, 2D:]

WHY IT IS ONE MATRIX AND NOT THREE. Mathematically `x @ [Wq | Wk | Wv]` is
exactly the three products side by side -- the fusion changes nothing about the
answer. What it changes is the shape of the work: one GEMM with a 3x wider N
instead of three GEMMs, so x is read once instead of three times and there is
one kernel launch instead of three. GPT-2 ships the checkpoint that way
(`c_attn.weight` is [768, 2304]) and every implementation since has copied it,
which is why `split(3, dim=-1)` appears in essentially every decoder.

HOW THIS DIFFERS FROM swiglu, which is the kernel next to it in shape. swiglu
also runs sibling matmuls off one x tile, but off two SEPARATE weight buffers.
Here the three weights are three COLUMN RANGES OF ONE BUFFER, and that is the
whole point of the file:

  swiglu      three tensors, three base pointers, three independent layouts
  this        three tensors, ONE base pointer, offsets N and 2N apart

So the backend has to answer whether three tiles cut out of one buffer at
different column offsets can each be given a lane axis, and whether those three
answers agree. They are siblings that share provenance, which is a stronger
constraint than siblings that merely share a consumer.

WHY THE SPLIT IS ALONG THE FREE AXIS, and why that is not an accident. The cut
runs along N, the matmul's output-column axis. Cutting along M -- the row axis,
which is the one pinned to the lanes -- would be a rebank, because rows live in
banks and a row range is a bank range. Real decoders never cut that way either:
the split is always over features, never over tokens. The hardware's cheap
direction and the model's natural direction happen to be the same one, which is
worth stating precisely because it is the kind of alignment that gets assumed.

WHAT THE CONSUMER IS. Attention proper is `attention.py`'s job -- this kernel
ends at `q * k + v` instead, which is meaningless as a model but forces all
three slices live at once and joins them elementwise. If the three lane axes
disagreed, that join is where it would show.

torch equivalent: q, k, v = (x @ w).split(N, dim=1); out = q * k + v

GPT-2: three sibling matmuls whose weights are three COLUMN RANGES OF ONE BUFFER -- siblings that share provenance, not just a consumer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 128, 64, 128           # tokens, head width, model width
BM, BN, BK = 128, 64, 128
N3 = 3 * N                       # the fused weight's output width


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def qkv_split_kernel(x_ptr, w_ptr, o_ptr,
                         N: tl.constexpr, K: tl.constexpr, N3: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr,
                         BK: tl.constexpr):
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        q = tl.zeros((BM, BN), dtype=tl.float32)
        k_ = tl.zeros((BM, BN), dtype=tl.float32)
        v = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in range(0, K, BK):
            a = tl.load(x_ptr + offs_m[:, None] * K + (kk + offs_k)[None, :])
            # The three weight tiles: same base pointer, same row stride N3,
            # column offsets 0 / N / 2N. This is the split -- it happens at LOAD
            # time, not after the matmul, which is what a real implementation
            # does too once the GEMM is tiled.
            wq = tl.load(w_ptr + (kk + offs_k)[:, None] * N3
                         + offs_n[None, :])
            wk = tl.load(w_ptr + (kk + offs_k)[:, None] * N3
                         + (N + offs_n)[None, :])
            wv = tl.load(w_ptr + (kk + offs_k)[:, None] * N3
                         + (2 * N + offs_n)[None, :])
            q += tl.dot(a, wq)
            k_ += tl.dot(a, wk)
            v += tl.dot(a, wv)

        tl.store(o_ptr + offs_m[:, None] * N + offs_n[None, :], q * k_ + v)

    return qkv_split_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K, generator=g, dtype=torch.float32) * 0.1,
            "w": torch.randn(K * N3, generator=g, dtype=torch.float32) * 0.1}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(M, K)
    qkv = x @ inputs["w"].reshape(K, N3)
    q, k, v = qkv.split(N, dim=1)
    return {"o": (q * k + v).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="qkv_split_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr", "K": "constexpr", "N3": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"N": N, "K": K, "N3": N3, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("x", IN, "float32", (M * K,)),
          Arg("w", IN, "float32", (K * N3,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
