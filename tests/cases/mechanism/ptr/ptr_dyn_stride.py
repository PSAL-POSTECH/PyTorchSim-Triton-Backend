"""A matmul whose leading dims are RUNTIME scalars, not `constexpr`.

This is tests/cases/smoke/gemm.py with `N` and `K` moved out of `constexprs` into ordinary
i32 kernel arguments. That one change is what the README's "스칼라 인자는
constexpr로 둔다" note was about: PtrAnalysis leaves the leading-dim stride as
`kDynamic` plus an SSA operand either way, but when the operand folds to an
`arith.constant` the backend can bake it into the DMA descriptor global, and
when it does not there is nowhere static to put it.

Before the runtime-stride path existed this raised

    dynamic stride is not a compile-time constant -- declare the tensor shape
    constexpr

Now the unfolded stride rides on the transfer as an i64 operand (`dyn_stride_axes`)
and stage 4 stores it into mm_stride[axis] -- i32 words 12+2*axis / 13+2*axis,
byte 48+8*axis -- of a per-op descriptor, the same runtime-patch path the mask
clamp and a runtime `other=` use.

WHY BOTH N AND K
  They exercise different operands: K is A's row stride and N is B's row stride
  AND C's row stride, so one runtime value reaches two transfers of opposite
  direction (MVIN and MVOUT). A backend that only patched the load side passes
  the A tile and fails on C.

WHAT PROVES IT FIRED (04-transformed.mlir)
      dram_stride = [0, 1], dyn_stride_axes = [0]
  -- the 0 in slot 0 is the placeholder, not a real stride. Compare gemm.py,
  which has `dram_stride = [256, 1]` and no dyn_stride_axes at all because 256
  folded.

M stays constexpr: it is a tensor extent, not a stride, and the harness needs a
static element count for the flat DRAM view (the memref args are unranked, so
`arg_numels` comes from the spec rather than the IR).

torch equivalent: torch.matmul(a, b).

gemm with the leading dims as runtime i32 args -- the runtime dram_stride path. Was a hard _Fatal before.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 512, 512, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_dyn_stride_kernel(a_ptr, b_ptr, c_ptr, N, K,
                              M: tl.constexpr, BM: tl.constexpr,
                              BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return ptr_dyn_stride_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M, K, generator=g, dtype=torch.float32),
            "b": torch.randn(K, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"], inputs["b"])}


SPEC = KernelSpec(
    name="ptr_dyn_stride_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "N": "i32", "K": "i32", "M": "constexpr", "BM": "constexpr",
               "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "BM": BM, "BN": BN, "BK": BK},
    args=[Arg("a", IN, "float32", (M, K)),
          Arg("b", IN, "float32", (K, N)),
          Arg("c", OUT, "float32", (M, N))],
    grid=(M // BM, N // BN), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("N", "int32_t"), ("K", "int32_t")],
           "scalar_values": {"N": N, "K": K}},
    rtol=1e-3, atol=1e-3,
)
