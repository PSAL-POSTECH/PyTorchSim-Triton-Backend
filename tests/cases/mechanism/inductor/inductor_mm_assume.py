"""`tl.assume` becomes `llvm.intr.assume`, which triton-shared-opt cannot parse.
It never reaches it: the text drop in front of the linalg lowering drops the op on the kernel's own Triton
IR, above the linalg lowering. Hand the raw ttir over instead and this is what comes
back, measured today:

    01-ttir.mlir:17:5: error: Dialect `llvm' not found for custom op 'llvm.intr.assume'
    note: Available dialects: arith, bufferization, builtin, cf, func, linalg,
          math, memref, ptr, scf, tensor, tptr, tt, tts, ttx, ub

Deleting only those lines from the ttir makes the same stage-2 command succeed
and produce a `linalg.matmul`:

    $ grep -v llvm.intr.assume 01-ttir.mlir > noassume.mlir
    $ triton-shared-opt --triton-to-structured --cse --canonicalize \\
        --triton-to-unstructured --triton-arith-to-linalg=tensor-ptr-to-linalg=true \\
        --triton-ptr-to-memref --mlir-print-op-generic noassume.mlir -o out.mlir
    $ grep -c linalg.matmul out.mlir
    1

`tl.assume` is a pure optimisation hint, so dropping it is free. It is not
exotic either: Inductor's mm template emits it unconditionally
(torch/_inductor/kernel/templates/triton_mm.py.jinja, the two lines after the
GROUP_M swizzle), so every matmul that comes from that template carries it.

This kernel is the smallest thing that shows it -- the surrounding matmul is
tests/cases/smoke/gemm.py, which passes today. Remove the two tl.assume lines and this
passes too.

tests/cases/smoke/gemm.py plus two tl.assume lines. They lower to llvm.intr.assume, a dialect stage 2 does not load, and nothing else in the kernel is new.
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
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
