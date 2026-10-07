"""AssertConverter -- BOTH arms, plus the assert-to-cf pass option.

Covers three census ids that nothing else in the set reaches:

  assert-scalar-cf        TritonArithToLinalg.cpp:60 / ConversionPatterns.hpp:819
  assert-tensor-generic   TritonArithToLinalg.cpp:60 / ConversionPatterns.hpp:823
  opt-assert-to-cf        Passes.td:22  (assertToCf, default true)

AssertConverter has exactly two arms and they are mutually exclusive per
`tt.assert`, so one arm per assert: `tl.device_assert` on a SCALAR i1 takes the
first, on a TENSOR i1 the second. opt-assert-to-cf comes free -- its only
observable is that AssertConverter got registered at all, which is precisely
what the two arms firing proves.

This is a POSITIVE kernel. Unlike mask_assert_revcmp.py / mask_assert_expand3d.py
-- whose names also say "assert" but mean a C++ assert killing triton-shared-opt
-- nothing here fails: every pass succeeds, 98-full == 99-composite, and both
conditions are TRUE at runtime (`n` is 128, and randn produces no NaNs). It is
still a compile-time probe like the rest of tests/cases/: the interesting
artifact is 03-arith-linalg.mlir, not a Spike run.

WHY IT NEEDS debug=True
-----------------------
`tl.device_assert` is COMPILED OUT unless the frontend debug flag is set.
triton/language/semantic.py:1850-1852:

    def device_assert(self, cond, msg, mask):
        if not self.builder.options.debug:
            return

so with the harness's default options no `tt.assert` is ever built, and
AssertConverter is dead for the whole set -- exactly the same shape of gap as
`supported_fp8_dtypes` was. The flag arrives via SPEC.extra["ttir_options"]
below, which pytorchsim_triton_compiler/lowering/triton_ttir.py merges into its `backend.parse_options({...})`
dict.

WHY THE FLAG MUST BE PER-KERNEL, NOT GLOBAL
-------------------------------------------
debug=True also turns on Triton's implicit i32 overflow checks, which staple an
extra `tt.assert ... "int32 overflow detected for operation <op>"` onto every
i32 multiply and add -- i.e. onto the offset arithmetic of essentially every
other kernel here. Measured on an earlier draft of this very file, whose only
difference was `offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)`: 01-ttir
grew from 2 `tt.assert` to 4 (an i1 one for the `mul`, a tensor<128xi1> one for
the `add`) and 03-arith-linalg grew from 33 lines to 92. Setting debug globally
would perturb every census kernel's IR. Hence the SPEC-level hook, and hence
this file keeps its own index math to a bare `tl.arange` so the only two
`tt.assert`s in the module are the two deliberate ones, one per arm.

MEASURED
--------
01-ttir.mlir, the whole assert population:
    tt.assert %0, "n must be positive" : i1
    tt.assert %1, "no nans" : tensor<128xi1>

03-arith-linalg.mlir:10-11, assert-scalar-cf -- a single cf.assert, message
wrapped in the converter's llvm::formatv("Assertion `{0}` failed"), and the
tt.assert erased:
    %4 = arith.cmpi sgt, %arg2, %c0_i32 : i32
    cf.assert %4, "Assertion `n must be positive` failed"

03-arith-linalg.mlir:18-22, assert-tensor-generic -- note what is NOT there: no
`->` result type (TypeRange{}) and no `outs(...)` clause (ValueRange{}). A
linalg.generic with zero results and an empty outs is unique to this arm in the
entire pass; the body is a cf.assert per element plus a bare linalg.yield:
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]}
                   ins(%6 : tensor<128xi1>) {
    ^bb0(%in: i1):
      cf.assert %in, "Assertion `no nans` failed"
      linalg.yield
    }

opt-assert-to-cf, both directions. Default (assertToCf=true, which is what the
composite pipeline leaves it at): 01-structured and 02-unstructured carry
2 tt.assert / 0 cf.assert, then 03/04/05/98/99 all carry 0 tt.assert /
2 cf.assert -- the converter ran, and its output survives to the end. Forced
off:
    triton-shared-opt --triton-arith-to-linalg='tensor-ptr-to-linalg=true \\
        assert-to-cf=false' work/assert_ops_kernel/01-ttir.mlir
exits 0 with both `tt.assert` verbatim and zero cf.assert -- the pass simply
declared them legal, TritonArithToLinalgPass.cpp:185-187
`if (!assertToCf) { target.addLegalOp<triton::AssertOp>(); }`.

The whole step-by-step pipeline succeeds and 98-full.mlir == 99-composite.mlir
byte for byte, so nothing about this kernel is a special case for
tools/lower_steps.py.

DO NOT "SIMPLIFY" EITHER CONDITION
----------------------------------
`n` must stay a RUNTIME i32 argument. As a constexpr, `n > 0` folds in Python
and no tt.assert is built at all, silently deleting assert-scalar-cf. And the
tensor condition must stay float-typed: `x == x` is `arith.cmpf oeq` on
tensor<128xf32>, which no canonicalizer may fold to true (NaN), so it reaches
triton-shared as a live tensor<128xi1>.

AssertConverter census (compile-time probe). The ONLY kernel with ttir_options -- see the docstring for why debug=True cannot be set globally.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    """Lazy: no triton import at module scope. See pytorchsim_triton_compiler/contract/kernel_spec.py."""
    import triton
    import triton.language as tl

    @triton.jit
    def assert_ops_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        # No pid arithmetic on purpose -- see the module docstring. debug=True
        # decorates i32 arith mul/add with an overflow tt.assert, but not
        # tt.make_range and not tt.addptr, so a bare arange fed straight into a
        # pointer add adds no assert of its own.
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)

        # SCALAR i1 -> ConversionPatterns.hpp:819 -> a plain cf.assert.
        tl.device_assert(n > 0, "n must be positive")

        # TENSOR i1 -> ConversionPatterns.hpp:823 -> linalg.generic with no
        # results and no outs, one cf.assert per element.
        tl.device_assert(x == x, "no nans")

        tl.store(o_ptr + offs, x * 2.0)

    return assert_ops_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # randn never produces NaN, so `x == x` holds elementwise and the tensor
    # assert is satisfied -- this kernel is not supposed to trap.
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    # grid is (1,) and BLOCK == N, so the single program writes the whole buffer.
    return {"o": inputs["x"] * 2.0}


SPEC = KernelSpec(
    name="assert_ops_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # ttir_options is merged into backend.parse_options() by
    # pytorchsim_triton_compiler/lowering/triton_ttir.py; without debug=True this file emits no tt.assert and
    # covers nothing. `n` is additionally a runtime scalar, so the generated C
    # driver (pytorchsim_triton_compiler/lowering/wrapper.py:_scalars) must be told to pass it -- otherwise the
    # trailing grid/pid i32s slide into its slot and `n > 0` reads garbage.
    extra={"ttir_options": {"debug": True},
           "scalar_args": [("n", "int32_t")],
           "scalar_values": {"n": N}},
    expect="refuse",
    expect_why="vectorize refuses first: `cf.assert in a linalg body is "
               "not known elementwise`. The AssertConverter diagnostics this "
               "file was written to probe are no longer what stops it "
               "(census 2026-09-04)",
)
