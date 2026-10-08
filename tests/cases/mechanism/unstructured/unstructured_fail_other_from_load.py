"""NEGATIVE: `other` that getScalarValue cannot walk down to a scalar.

Two loads, both with `other=oth` where oth is produced by a tt.load:

  * load #1 has a STRUCTURED ptr and a STRUCTURED mask, so rewriteLoadOp gets
    past the ptr check (:1774), the scalar check (:1782) and MaskAnalysis
    (:1796) and actually reaches the `other` check at PtrAnalysis.cpp:1811.
  * load #2 is the original gather shape (data-dependent ptr AND mask): it
    bails at "MaskAnalysis failed" long before the `other` check.

Both survive triton-to-structured as tt.load, so triton-to-unstructured then
calls tts::utils::getScalarValue on `other` again and fails at
TritonToUnstructuredPass.cpp:489.

Claims: load-other-unsupported, gsv-unsupported-op-fail,
        t2u-gather-other-unparseable-fail, t2u-pass-failure-warning-only

THIS KERNEL IS EXPECTED TO FAIL THE PIPELINE; the bail IS the observable. All
six passes of the linalg lowering still exit 0 (the failure is diagnostic-only: two
"other value used in masked load produced by unsupported instruction" errors,
one "cannot parse `other` value for load", and the module-level "Cannot
transform tensor of pointers into a single base pointer with tensor of
offsets" warning), but the unrewritten tensor-of-pointer ops fall through to
the tptr.* fallback and run.py then dies in triton-compile because the
backend MLIR has no `tptr` dialect. `reference()` below is never executed; it
is kept exact so it stays usable if that gap ever closes.

NEGATIVE kernel: the pass must BAIL. Stage 2 fails; not runnable end to end.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fail_other_from_load_kernel(x_ptr, i_ptr, t_ptr, o_ptr,
                                    BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        idx = tl.load(i_ptr + offs)
        oth = tl.load(t_ptr + offs)              # defining op is a load

        # structured ptr + structured mask: rewriteLoadOp reaches the `other`
        # check and fails there (PtrAnalysis.cpp:1812)
        v0 = tl.load(x_ptr + offs, mask=offs < 64, other=oth)

        # gather ptr + data-dependent mask: MaskAnalysis fails first, the load
        # falls through to --triton-to-unstructured, and getScalarValue fails
        # there instead (TritonToUnstructuredPass.cpp:489)
        v1 = tl.load(x_ptr + idx, mask=idx < 64, other=oth)

        tl.store(o_ptr + offs, v0 + v1)

    return fail_other_from_load_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(N, generator=g, dtype=torch.float32),
        "i": torch.randint(0, N, (N,), generator=g, dtype=torch.int32),
        "t": torch.randn(N, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    # Never evaluated -- run.py stops at stage 2 (see the module docstring).
    import torch
    x, i, t = inputs["x"], inputs["i"], inputs["t"]
    offs = torch.arange(N, dtype=torch.int32)
    v0 = torch.where(offs < 64, x, t)          # tl.load(x_ptr + offs, mask=offs < 64, other=t)
    v1 = torch.where(i < 64, x[i.long()], t)   # tl.load(x_ptr + idx,  mask=idx  < 64, other=t)
    return {"o": v0 + v1}


SPEC = KernelSpec(
    name="fail_other_from_load_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "t_ptr": "*fp32",
               "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("i", IN, "int32", (N,)),
          Arg("t", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse",
    expect_why="OUR refusal is what stops it: `a tensor tt.load whose other= "
               "varies per element is not wired; the gather's fill is a "
               "single value`. Same message as ptr_gather_other_varies "
               "(census 2026-09-04)",
)
