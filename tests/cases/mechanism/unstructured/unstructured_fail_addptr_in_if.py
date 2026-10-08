"""NEGATIVE: t2u-addptr-in-scf-if-fail -- tt.addptr nested inside an scf.if.

THIS KERNEL IS EXPECTED TO FAIL THE PIPELINE; the failure IS the observable.
--triton-to-unstructured refuses to rewrite the load, the raw tt.addptr /
tensor-of-pointer types survive into 02-unstructured.mlir, and run.py then dies
in triton-compile because the tptr.* fallback ops have no backend dialect.
`reference()` below is therefore never executed.

TritonToUnstructuredPass.cpp:324. `addptr->getParentOfType<scf::IfOp>()` is the
guard: the pass cannot prove the pointers yielded by both branches share a base,
so it refuses. Silent failure() -> module warning at :577 ("Cannot transform
tensor of pointers into a single base pointer with tensor of offsets"), no
emitError, and zero tts.gather in the output.

The mask has to be data dependent (idx < 64, idx loaded from memory) or
PtrAnalysis folds the whole thing into a structured tts.make_gather_scatter_tptr
in the previous pass and no tt.addptr survives to be looked at.

torch equivalent: x[idx + 128] where idx < 64, else 0  (with FLAG = 1).

NEGATIVE kernel: --triton-to-unstructured must BAIL. Stage 2 fails; not runnable end to end.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
FLAG = 1        # runtime value of `flag`; pinned in SPEC.extra below


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fail_addptr_in_if_kernel(x_ptr, i_ptr, o_ptr, flag, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        idx = tl.load(i_ptr + offs)
        p = x_ptr + idx
        if flag > 0:
            p = p + BLOCK                  # tt.addptr inside scf.if
        tl.store(o_ptr + offs, tl.load(p, mask=idx < 64))

    return fail_addptr_in_if_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(2 * N, generator=g, dtype=torch.float32),
            "i": torch.randint(0, N, (N,), generator=g, dtype=torch.int32)}


def reference(inputs):
    """Never evaluated -- the pipeline dies in stage 2 (see the module docstring).

    Kept exact anyway, per lane j:
        o[j] = x[i[j] + (FLAG > 0 ? BLOCK : 0)]   if i[j] < 64
             = <no `other` operand>               otherwise, taken as 0.0
    `x` is 2*N long because the `+ BLOCK` bias is NOT covered by the mask, so
    the widest address the kernel can form is (N-1) + BLOCK = 255.
    """
    import torch
    idx = inputs["i"].to(torch.int64)
    bias = N if FLAG > 0 else 0
    return {"o": torch.where(idx < 64, inputs["x"][idx + bias],
                             torch.zeros((), dtype=torch.float32))}


SPEC = KernelSpec(
    name="fail_addptr_in_if_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "flag": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": N},
    args=[Arg("x", IN, "float32", (2 * N,)), Arg("i", IN, "int32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("flag", "int32_t")],
           "scalar_values": {"flag": FLAG}},
    expect="refuse",
    expect_why="TritonToUnstructured leaves a tensor of pointers that cannot bufferize",
)
