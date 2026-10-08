"""Inductor's mm template wraps BOTH offsets, so its store wraps two axes.

    offs_m = (pid_m * BM + tl.arange(0, BM)) % M      axis 0
    offs_n = (pid_n * BN + tl.arange(0, BN)) % N      axis 1

    a = load(a_ptr + offs_m[:,None]*K + (k+offs_k)[None,:])   axis 0 only
    b = load(b_ptr + (k+offs_k)[:,None]*N + offs_n[None,:])   axis 1 only
    store(c_ptr + offs_m[:,None]*N + offs_n[None,:], acc)     BOTH

WHAT IT PINS. `shape` has a slot per axis, and until 2026-08-31 only one of them
was ever filled: addState refused two states that both had a modulo, one
`hasModulo()` above a loop that was already per-axis. So the two loads passed
and the STORE -- the only access wrapping two axes -- fell to tts.scatter.

The addresses were right either way; what was lost is that the store is a
descriptor at all. It is `static_shape [4096, %N]` now, one wrap per axis.

WHERE IT COMES FROM. Inductor's mm template emits exactly these two lines:
    offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
so any matmul from that template reaches this. The alignment hints are not
needed to trigger it and neither is the GROUP_M pid swizzle.

With M = N = 64 and BM = BN = 32 the grid is 2x2, so `pid_m * 32 + [0, 32)` is
always in [0, 64) and `% 64` is the identity -- removing both `%` gives
tests/cases/smoke/gemm.py. THE WRAP IS INERT HERE ON PURPOSE: the file is about
where the fact is written down, not about the arithmetic.

tests/cases/smoke/gemm.py with a no-op modulo on both offsets, so the store wraps two axes -- Inductor's mm template verbatim.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = N = K = 64
BM = BN = 32
BK = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mm_mod_only(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        # The two lines under test. Inductor's mm template emits exactly these.
        offs_m = (pid_m * BM + tl.arange(0, BM)) % M
        offs_n = (pid_n * BN + tl.arange(0, BN)) % N
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return mm_mod_only


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M * K, generator=g, dtype=torch.float32),
            "b": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    a = inputs["a"].reshape(M, K)
    b = inputs["b"].reshape(K, N)
    return {"c": (a @ b).reshape(-1)}


SPEC = KernelSpec(
    name="mm_mod_only",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "float32", (M * K,)),
        Arg("b", IN, "float32", (K * N,)),
        Arg("c", OUT, "float32", (M * N,)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    # The store is the access that wraps two axes; a scatter here means the
    # per-axis merge went back to refusing the pair.
    expect_ir={"03-linalg.mlir": {"tts.scatter": 0, "tts.make_tptr": 3}},
)
