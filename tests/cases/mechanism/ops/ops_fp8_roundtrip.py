"""FpToFpConverter -- the only tt.fp_to_fp in the whole kernel set.

Split out of the old tests/cases/ops.py.  It gets a file to itself because
it is the only case in the split that needs a FRONTEND option: see "WHY IT NEEDS
ttir_options" below.

COVERS  (line numbers are the triton-shared WORKING TREE, source root
         /workspace/triton_shared/)

  fptofp-truncate   lib/Conversion/TritonArithToLinalg/
                    TritonArithToLinalg.cpp:74
      FpToFpConverter when getBitWidth(operand) > getBitWidth(result)
      (ConversionPatterns.hpp:947) -> arith.truncf.  The rounding-mode
      attribute is DROPPED, not forwarded; the converter asserts it is not RTZ
      (RTNE is the default when the attribute is absent, which is this case).

  fptofp-extend     TritonArithToLinalg.cpp:74
      the fall-through when operandWidth <= resultWidth
      (ConversionPatterns.hpp:954) -> arith.extf.

WHY BOTH IN ONE FILE
  They are the two halves of one if/else in one converter, and the natural
  witness is a ROUND TRIP: you cannot store an fp8 tensor (see below), so the
  only way to observe the truncate is to extend the result back and store
  that.  The pair is therefore a single indivisible case.

WHY IT NEEDS ttir_options
  triton's own AST walk rejects fp8 before triton-shared ever sees the module:

      type fp8e5 not supported in this architecture.
      The supported fp8 dtypes are ()

  The backend's Options.supported_fp8_dtypes must list it first.  This SPEC
  therefore carries

      extra={"ttir_options": {"supported_fp8_dtypes": (...)}}

  which pytorchsim_triton_compiler/lowering/triton_ttir.py's `emit` merges into backend.parse_options().  The
  worker happens to set the same tuple as its own default today, so this file
  would compile without the entry -- it is stated explicitly anyway, because
  this is the one kernel whose ability to produce ANY IR depends on that
  option, and a silent default is exactly the kind of thing that gets tidied
  away by someone who does not know a kernel depends on it.

WHY THERE IS NO fp8 BUFFER
  fp8 is COMPUTE-ONLY in this harness.  A kernel argument additionally needs
  entries in pytorchsim_triton_compiler/lowering/wrapper.py's C_TYPE/DTYPE_BYTES and verify.py's NP_DTYPE, and
  there is no C or numpy scalar type for f8E5M2 to round-trip a .raw file
  through.  So: f32 in, f8E5M2 in the middle, f32 out.

WHAT PROVES IT FIRED   work/ops_fp8_roundtrip_kernel/triton-shared-detail/

  01-structured.mlir has the pair, and note the rounding attribute is present on
  the truncating one and absent on the extending one:

      %5 = tt.fp_to_fp %4, rounding = rtne : tensor<128xf32> -> tensor<128xf8E5M2>
      %6 = tt.fp_to_fp %5 : tensor<128xf8E5M2> -> tensor<128xf32>

  03-arith-linalg.mlir -- one converter, two arms, and the rounding attribute is
  simply GONE from the truncf, exactly as the census says (dropped, not
  forwarded).  The whole module is 22 lines, of which these are the point:

      %2 = tensor.empty() : tensor<128xf8E5M2>
      %3 = linalg.generic ... ins(%1 : tensor<128xf32>)
                              outs(%2 : tensor<128xf8E5M2>) {
      ^bb0(%in: f32, %out: f8E5M2):
        %7 = arith.truncf %in : f32 to f8E5M2         <- fptofp-truncate
        linalg.yield %7 : f8E5M2
      } -> tensor<128xf8E5M2>
      %4 = tensor.empty() : tensor<128xf32>
      %5 = linalg.generic ... ins(%3 : tensor<128xf8E5M2>)
                              outs(%4 : tensor<128xf32>) {
      ^bb0(%in: f8E5M2, %out: f32):
        %7 = arith.extf %in : f8E5M2 to f32           <- fptofp-extend
        linalg.yield %7 : f32
      } -> tensor<128xf32>

  The f8E5M2 element type appearing at all is the second half of the proof: it
  is the only place in the kernel set where a sub-16-bit float type reaches
  linalg, and it only exists because of the SPEC.extra option below.

ARITHMETIC
  None at all -- the value is only re-typed.  The reference is torch's own
  float8_e5m2 round trip, which is bit-exact against arith.truncf's RTNE:
  fp8e5 (E5M2) keeps 2 explicit mantissa bits, so the output is visibly
  quantised and a silently-skipped truncate would NOT compare equal.

SCOPE
  IR-census kernel.  Both ids were confirmed by gdb line hits
  (the branch census) and the IR was read from tools/lower_steps.py.
  Like every file under tests/cases/ it does not reach run.py's numeric
  stages, so the bit-exactness claim above is about what arith.truncf MEANS, not
  about a comparison this harness has executed.

POSITIVE KERNEL: every triton-shared pass succeeds.

FpToFpConverter truncate + extend. The only tt.fp_to_fp in the set, and the only census kernel whose IR depends on a frontend option other than tests/cases/mechanism/ops/ops_assert.py. POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_fp8_roundtrip_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r)
        small = x.to(tl.float8e5)       # tt.fp_to_fp f32 -> f8E5M2  (truncate)
        back = small.to(tl.float32)     # tt.fp_to_fp f8E5M2 -> f32  (extend)
        tl.store(o_ptr + r, back)

    return ops_fp8_roundtrip_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"].to(torch.float8_e5m2).to(torch.float32)}


SPEC = KernelSpec(
    name="ops_fp8_roundtrip_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # Merged on top of pytorchsim_triton_compiler/lowering/triton_ttir.py's defaults. Without fp8e5 in this
    # tuple the frontend refuses `.to(tl.float8e5)` and no tt.fp_to_fp exists.
    # e4nv is deliberately absent: allow_fp8e4nv is a separate flag and the
    # nv-specific variants carry different bias conventions.
    extra={"ttir_options": {
        "supported_fp8_dtypes": ("fp8e5", "fp8e4b8", "fp8e4b15", "fp8e5b16")}},
    expect="refuse",
    expect_why="8-bit float is not on this backend's path, and that is a decision rather than a limitation waiting to lift. The SIMULATOR half has lifted -- spike carries E4M3/E5M2 and the array multiplies in it -- so a reader who sees the hardware working would otherwise read this refusal as stale. What remains is upstream LLVM or a fork of the RISC-V backend, and we are not carrying it. Hence refuse, not blocked: no amount of compiler progress changes the answer without someone deciding differently",
)
