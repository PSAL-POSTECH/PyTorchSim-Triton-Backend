"""`tl.assume`, which triton lowers into `llvm.intr.assume`.

Inductor's mm template emits it unconditionally
(torch/_inductor/kernel/templates/triton_mm.py.jinja, the two lines after the
GROUP_M swizzle), so every matmul that comes from that template carries it.
This kernel is tests/cases/smoke/gemm.py plus exactly those two lines.

WHAT IT PINS: that the assume never reaches the linalg lowering at all.

  the text drop in front of the linalg lowering DROPS it, on the kernel's own Triton IR. A result-less
  optimisation hint, and nothing here reads a bound, so removing it cannot
  change the answer -- see pipeline.py:_drop_noops.

  Then the comparison it tested goes too, collected by the canonicalize that
  pass runs once its last reader is gone.

Measured, per artifact: 01-ttir has the two assumes and the two `arith.cmpi`
they test; 02-ttclean, 03-linalg and 04-transformed have NONE of the four.

THERE IS NO SECOND LINE OF DEFENCE, and that is deliberate. triton-shared-opt
briefly REGISTERED the llvm dialect as well, so the op parsed and rode through
unmatched; the text drop made that registration unreachable and it was taken out
again. If the text drop stops running, this kernel does not get a slower answer -- it
gets `Dialect 'llvm' not found for custom op 'llvm.intr.assume'` at stage 3.

tests/cases/smoke/gemm.py plus two tl.assume lines, and nothing else in the kernel is new.
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
    def mm_assume(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        # The two lines under test. Inductor's mm template emits exactly these.
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return mm_assume


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M * K, generator=g, dtype=torch.float32),
            "b": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    a = inputs["a"].reshape(M, K)
    b = inputs["b"].reshape(K, N)
    return {"c": (a @ b).reshape(-1)}


SPEC = KernelSpec(
    name="mm_assume",
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
)
