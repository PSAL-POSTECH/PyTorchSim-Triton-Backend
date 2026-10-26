"""One float tl.dot whose K is constexpr, so there is NO enclosing K loop.

THE WITNESS FOR "a matmul does not need a loop to be a reduction".

The linalg lowering emits `linalg.matmul` directly when K fits one tile -- a constexpr
K leaves nothing to iterate. lower_to_vcix used to require an enclosing scf.for
to call the matmul a reduction and told the kernel author to write `K = 2 * BK`
so one would appear, which is the compiler's shape leaking into the source: the
op already declares the reduction, and the K stepping the array walks is the
n/k subtile nest that pass builds itself.

Worse than a refusal, the decline was SILENT (`return False`), so the matmul
went to convert-linalg-to-loops, which reads the lane-banked scratchpad
linearly. Measured before the fix: all 1024 elements wrong, no diagnostic.
lowering/passes/lower.run now fails if any linalg.matmul survives lower_to_vcix.

Deliberately minimal -- one dot, float, no mask, no epilogue -- because
ops_dot.py cannot witness this: it bundles an int8 dot, and the matrix
unit is float throughout, so that file is refused for a different reason.

WHAT `expect_ir` PINS, AND WHY IT IS HERE. The array's whole vocabulary, spelt as
the `vcix.*` DIALECT ops of the matrix unit, each a group of four registers (32 rows of
f32): 10 `vcix.iv` (four vmatpush of B, one vmatmul of A, each after its vmatprep),
three `vcix.v.i` read a vector back -- two pops and the lane index, which is the same
instruction. Spelt instead as an `llvm.call_intrinsic` naming
`llvm.riscv.sf.vc.*.se` -- which is what this backend emitted while the fork's ops
were unavailable -- both counts are 0. The ASSEMBLY IS IDENTICAL either way,
so no value and no instruction can witness this; the artifact is the only place
the claim lives. The counts do not move under PYTORCHSIM_COMPILE_BANK_VIEW.

one constexpr-K float dot, no enclosing loop

LAYOUT PIN. `expect_ir` holds `ttg.dot_op` in 05-layouts.mlir at 0; what that pins is the dot anchor
(tritongpu-pytorchsim-anchor-layouts) giving A, B and C each a lane layout of its own shape; the conversion's
dot_op (A replicated in every lane) is not this machine's.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 32
K = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_dot_constexpr_k_kernel(a_ptr, b_ptr, o_ptr, M: tl.constexpr, K: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])     # M x K
        b = tl.load(b_ptr + rk[:, None] * M + rm[None, :])     # K x M
        c = tl.dot(a, b)
        tl.store(o_ptr + rm[:, None] * M + rm[None, :], c)

    return ops_dot_constexpr_k_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(M * K, generator=g, dtype=torch.float32),
            "b": torch.randn(K * M, generator=g, dtype=torch.float32)}


def reference(inputs):
    a = inputs["a"].reshape(M, K)
    b = inputs["b"].reshape(K, M)
    return {"o": (a @ b).reshape(-1)}


SPEC = KernelSpec(
    name="ops_dot_constexpr_k_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "K": K},
    args=[Arg("a", IN, "float32", (M * K,)), Arg("b", IN, "float32", (K * M,)),
          Arg("o", OUT, "float32", (M * M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"05-layouts.mlir": {"ttg.dot_op": 0},
               "09-custom.mlir": {"vcix.iv": 10, "vcix.v.i": 3}},
    expect_ir_on={"tpuv3": {"09-custom.mlir": {"vcix.iv": 5, "vcix.v.i": 2}}},
)
