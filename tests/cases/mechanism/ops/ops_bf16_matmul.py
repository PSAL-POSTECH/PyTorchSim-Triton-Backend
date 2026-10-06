"""bf16 -- sixteen bits this machine has no instruction for, and no diagnostic.

It sits next to tests/cases/mechanism/ops/ops_fp8_roundtrip.py because what this file
is about is an ELEMENT TYPE the target cannot compute in, not the matmul it
happens to use to show it -- and the two are the two halves of the same pass,
pytorchsim_triton_compiler/elf/passes/p00_refuse_unsupported_dtypes.py. fp8 is caught by WIDTH (anything
narrower than an f16). bf16 is caught by NAME, and the reason it needs its own
rule is the reason it needs its own kernel: it is sixteen bits wide, exactly
like the fp16 this machine DOES have, so every width test in the backend waves
it through. (They do not share an `expect`; see the bottom of this docstring.)

WHAT IS ACTUALLY MISSING
  bf16 is an f32 truncated to its top 16 bits -- 8 exponent bits, 7 of mantissa
  -- and an IEEE half is 5 and 10. Same width, different format. The vtype does
  not carry a format: e16 on this target means IEEE half, because Zfh is the
  only 16-bit float the ISA names. There is no bf16 extension at all -- spike
  rejects zfbfmin, zvfbfmin and zvfbfwma outright ("unsupported extension"), so
  there is not even a widening convert to reach f32 through, and llc's
  -mattr has no bf16 either.

WHY THAT IS WORSE THAN AN ERROR, WHICH IS THE WHOLE POINT OF THIS FILE
  Nothing in the pipeline objects. Measured on the tree with only the bf16 arm
  of the refusal removed (HEAD minus that hunk, everything else identical):

      stage 3  adapt     ok      the pytorchsim-triton-compiler passes are happy
      stage 4  lower     ok      vcix -> DMA -> LLVM, no complaint
      stage 5  binary    llc exit 0, a valid 3592-byte object file
      stage 6  spike     RUNS, and returns a full C tile of numbers

  and the object file's vector instructions are

      16 vle16.v   16 sf.vc.iv   16 sf.vc.v.i   16 vfadd.vv   16 vle32.v ...

  which is byte-for-byte the shape of the fp16 kernel in
  tests/cases/mechanism/mixed/mixed_gemm_fp16.py. ZERO bf16 instructions, because
  there are none to emit -- the bf16 words are loaded as 16-bit words, handed to
  the array at SEW=16, and the array decodes them as IEEE halves. The numbers
  that come back are computed from a different format than the one the kernel
  declared:

      max_abs_err 182.775, max_rel_err 21335.7
      16342 of 16384 elements outside tolerance
      worst: got 218.53635 where the reference says 35.761185

  A backend that answers 218 for 35 without a word from any stage is the failure
  this refusal exists to prevent. (Reaching stage 6 at all took three lines of
  control-side instrumentation -- wrapper.py's C_TYPE and DTYPE_BYTES and the
  input dump have no bfloat16 entry, so the unpatched control stops at stage 5
  on `KeyError: 'bfloat16'`. That is a missing dict entry, not a statement about
  the machine, and filling it in is a one-line change any passer-by might make.)

WHY THE REFUSAL IS NOT IN lower_to_vcix, WHERE THE LIE IS VISIBLE
  Because a bf16 ELEMENTWISE kernel lies in exactly the same way and never
  reaches a matmul: vfadd.vv at e16 is IEEE-half addition whatever the bits
  meant. The fact is about the element type, so it is stated once, at the front,
  while the type is still the one the kernel wrote -- one pass further in and
  this backend has already built selects, address arithmetic and math
  derivations in it, and the message would name ops nobody asked for instead of
  the dtype that caused them.

WHY expect="blocked" WHERE THE fp8 KERNEL NEXT DOOR SAYS "refuse"
  They share a pass and they do NOT share an answer, and the difference is
  whether anyone has to DECIDE anything. 937d86f moved ops_fp8_roundtrip to
  "refuse" because there is no standard 8-bit float extension to wait for: the
  only one that exists is this project's own non-standard Zvfp8, so closing that
  gap means someone choosing to carry a fork of the RISC-V backend, and no
  amount of upstream progress produces it on its own.

  bf16's extensions are NAMED in the standard -- Zfbfmin, Zvfbfmin, Zvfbfwma --
  and the only thing wrong is that the trees in front of us do not have them
  yet. Measured on the spike this harness runs (pytorchsim_triton_compiler/toolchain.SPIKE):

      --isa=rv64gcv_zfh_zvfbfmin  error: unsupported extension: zvfbfmin
      --isa=rv64gcv_zfh_zfbfmin   error: unsupported extension: zfbfmin

  Nobody has to decide to want that; it arrives when spike and llc grow what the
  spec already names, and on that day this kernel simply starts working. That is
  what "blocked" means here and why the two files differ: a "refuse" that starts
  passing is a regression to investigate, a "blocked" that starts passing is
  news, and the runner says so and asks for the marker back. Nothing in this
  file should be read as bf16 being unwelcome.

WHAT WOULD BREAK IT, AND THE ONE THING THAT WOULD NOT
  Deleting the `_is_bf16` arm, or rewriting the refusal to test width alone --
  16 is not < 16, so bf16 walks straight through a width bound. Also: moving the
  refusal below anything that rewrites element types, which would let the module
  reach the pass with the bf16 already laundered into an i16 or an f32.

  AND THE HONEST LIMIT, because a reader deserves it. `expect="blocked"` never
  fails a sweep -- pipeline.py returns True at whatever stage a blocked kernel
  stops at -- so if the bf16 arm were deleted tomorrow this line would still
  read PASS, just blocked further down the pipeline (at stage 5's KeyError
  instead of at stage 3's refusal). The sweep VERDICT is not the evidence here;
  WHERE it stops is, and the transcript is where that shows. What this file does
  buy is that the failure has a name and a place: without it there is no bf16
  anywhere in tests/cases/, and the day someone widens the wrapper tables
  the whole thing above happens with nothing in the suite standing where the
  numbers come out.

SHAPE
  The smallest matmul this backend accepts: one 128x128 tile, K spanning two BK
  tiles so the reduction loop survives triton's unroller. It is deliberately not
  the fp16 kernel's shape -- nothing here is measured, it only has to be a
  well-formed matmul that a bf16 value flows through.

bf16 operands with an f32 accumulator. The only bf16 in the suite. It is ALLOWED, by decision: the values are wrong and the cycles are not. Measured -- vle16 / vfadd / sf.vc.*, no bf16 instruction anywhere, and it answers 218 where the reference says 35.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 128, 128, 256
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_bf16_matmul_kernel(a_ptr, b_ptr, c_ptr,
                               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        # The accumulator is f32 on purpose: an all-bf16 dot would be refused
        # for its C operand too, and the file would no longer be able to say
        # which operand the refusal fired on. This one is refused for A and B.
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return ops_bf16_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32).to(torch.bfloat16),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.bfloat16),
    }


def reference(inputs):
    import torch
    # Never reached while the block stands, and written anyway: it is the
    # reference the numbers above were wrong AGAINST, and the day Zvfbfmin lands
    # it is what the kernel has to match. Widened before the matmul because that
    # is what the array does -- multiply in the input format, accumulate in
    # float.
    return {"c": torch.matmul(inputs["a"].to(torch.float32),
                              inputs["b"].to(torch.float32))}


SPEC = KernelSpec(
    name="ops_bf16_matmul_kernel",
    kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "bfloat16", (M, K)),
        Arg("b", IN, "bfloat16", (K, N)),
        Arg("c", OUT, "float32", (M, N)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    # bf16 keeps 8 significant bits, so an eventual real bf16 path would land
    # nowhere near the fp16 kernel's 1e-3. Stated for the day the block lifts.
    rtol=1e-2, atol=1e-2,
    expect="open",
    expect_why="bf16 runs for CYCLES, not for values -- the ISA has no bf16, so "
               "e16 decodes the bits as an IEEE half and 16342 of 16384 elements "
               "land outside tolerance, while the instruction mix, the widths "
               "and the traffic are a real bf16 kernel's",
)
