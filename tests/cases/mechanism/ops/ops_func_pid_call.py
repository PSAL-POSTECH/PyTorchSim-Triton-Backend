"""Everything pass 4 does at FUNCTION level: tt.func -> func.func, the six
appended program-info arguments, and tt.call -> func.call.

Split out of the old tests/cases/ops.py.  These five ids are one mechanism
seen from five angles, and they are genuinely coupled rather than merely
adjacent: addProgramInfo widens every tt.func by six i32 arguments, the two
GetProgramID/GetNumPrograms converters are the only readers of those arguments,
and CallConverter's sub-branch exists SOLELY to re-supply them at a call site
whose tt.call was built before the widening.  A file with only the pids, or only
the call, could not show that.

COVERS -- 5 branch ids.  Line numbers are the triton-shared WORKING TREE,
source root /workspace/triton_shared/.

  opt-pids-to-func-args    include/triton-shared/Conversion/
                           TritonArithToLinalg/Passes.td:16
      Option<"pidsToFuncArgs", default=true>.  Stage 3 leaves it at the
      default (pipeline.py's TTSHARED_PIPELINE sets no such flag), so
      addProgramInfo (TritonArithToLinalgPass.cpp:52) appends six i32 args and
      registers the two converters below.  A .td option, so gdb cannot break on
      it -- it is verified by the emitted argument list, quoted below.

  opt-tt-to-func-func      Passes.td:18
      Option<"ttToFuncFunc", default=true>.  NOT a RewritePattern at all and no
      ConversionTarget entry: a post-conversion walk
      (TritonArithToLinalgPass.cpp:208-245) clones each tt.func into a func.func
      and rewrites tt.return -> func.return.  Also .td, also verified by IR.

  pid-to-func-arg          lib/Conversion/TritonArithToLinalg/
                           TritonArithToLinalg.cpp:53
      GetProgramIDConverter, registered only `if (pidsToFuncArgs)`.  Emits NO
      op: pure SSA substitution with func argument #(numArgs - 3 + axis), one of
      the LAST three appended i32s.  Asserts axis < 3.

  numprograms-to-func-arg  TritonArithToLinalg.cpp:53
      GetNumProgramsConverter, same guard, argument #(numArgs - 6 + axis) --
      the FIRST three of the six.

  call-func-call           TritonArithToLinalg.cpp:67
      CallConverter, unconditional -> func.call with the same callee symbol.
      The interesting half is the sub-branch at ConversionPatterns.hpp:888-908:
      when the parent is a triton::FuncOp and the callee now needs MORE inputs
      than the call supplies, the missing TRAILING arguments are copied from the
      parent's own block arguments.  Here that is the normal case, not an exotic
      one, because addProgramInfo widened BOTH functions while the tt.call still
      carried its single original operand.

WHAT PROVES IT FIRED   work/ops_func_pid_call_kernel/triton-shared-detail/

  01-structured.mlir has two tt.func, both with the ORIGINAL argument lists, and
  the two program-info ops still standing.  (The callee's real symbol is
  mangled with the defining module -- shortened to @_bias here and below.)

      tt.func public @ops_func_pid_call_kernel(%arg0: !tt.ptr<f32>,
                                               %arg1: !tt.ptr<f32>)
        %0 = tt.get_program_id x : i32
        %1 = tt.get_num_programs x : i32
        %13 = tt.call @_bias(%12) : (f32) -> f32
        tt.return
      tt.func private @_bias(%arg0: f32) -> f32 attributes {noinline = true}

  03-arith-linalg.mlir: both are func.func, both widened by exactly six i32s
  (opt-tt-to-func-func + opt-pids-to-func-args), and the two ops are GONE --
  replaced, not lowered:

      func.func @ops_func_pid_call_kernel(%arg0: !tt.ptr<f32>,
          %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32,
          %arg5: i32, %arg6: i32, %arg7: i32)
      func.func private @_bias(%arg0: f32, %arg1: i32, %arg2: i32,
          %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) -> f32

  With numArgs == 8 the census arithmetic reads directly off the IR: the
  num_programs value is #(8-6+0) = %arg2 and the program id is #(8-3+0) = %arg5.
  Both appear bare, with no defining op:

      %0  = arith.muli %arg5, %c128_i32 : i32         <- pid-to-func-arg
      %10 = arith.addi %arg5, %arg2 : i32             <- both, side by side
      return                                          <- tt.return rewritten

  and the call carries SEVEN operands where the source had one -- the six
  trailing program-info args copied from the parent, which is the
  ConversionPatterns.hpp:888-908 sub-branch:

      %5 = call @_bias(%4, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
             : (f32, i32, i32, i32, i32, i32, i32) -> f32

  The callee body itself is untouched -- three ops and a `return %1 : f32` --
  so the six new arguments there are genuinely unused.  That is the point:
  addProgramInfo widens EVERY tt.func unconditionally, and CallConverter's job
  is only to make the call site agree with it.

WHY @triton.jit(noinline=True)
  It is the only way a tt.call survives to the linalg lowering at all: triton's own
  ttir pipeline runs the inliner, so an ordinary jit helper is gone long before
  pass 1.  That decorator is setup no other case in the split wants, which is
  why the call lives here rather than in a general elementwise file -- but it
  belongs with the pids, because without addProgramInfo the call would take the
  boring arity-matches arm.

WHY THE GRID IS (2,)
  With one program the pid would be a compile-time 0 and a wrong argument index
  would still produce the right numbers.  Two programs make pid (0/1) and
  num_programs (2) distinct, so picking the wrong appended slot is a visible
  numerical failure and not just a silent IR difference.

WHY THE CALLEE TAKES A SCALAR
  A tensor-valued callee would drag tensor conversion into the callee body and
  bury the widened-argument story.  One f32 in, one f32 out keeps @_bias to
  three ops, so the seven-operand func.call above is readable at a glance.

ARITHMETIC
  o[pid*128 + i] = x[pid*128 + i] + (3*x[0] + 1) + (pid + 2).  Exact.

SCOPE
  IR-census kernel.  The three code ids (pid-to-func-arg,
  numprograms-to-func-arg, call-func-call) were confirmed by gdb line hits with
  the branch census.  The two `opt-` ids are TableGen options with no
  line to break on, so they are verified by the argument lists and the func.func
  spelling quoted above -- that IS the measurement for a .td entry.  Like every
  file under tests/cases/ this does not reach run.py's numeric stages.

POSITIVE KERNEL: every pass of the linalg lowering succeeds.

Function-level pass-4 plumbing: tt.func -> func.func, addProgramInfo's six appended i32 args, GetProgramID/GetNumPrograms substitution, and CallConverter re-supplying those args at the call site. POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 128
GRID = 2
N = GRID * BLOCK


def kernel():
    import triton
    import triton.language as tl

    @triton.jit(noinline=True)   # the ONLY way a tt.call reaches the linalg lowering
    def _bias(a):
        return a * 3.0 + 1.0

    # triton resolves a callee from the DEFINING function's globals, and this
    # one is a local of kernel(); publish it so the @triton.jit body can see it.
    globals()["_bias"] = _bias

    @triton.jit
    def ops_func_pid_call_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)          # tt.get_program_id   -> %arg(N-3)
        npr = tl.num_programs(0)        # tt.get_num_programs -> %arg(N-6)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        b = _bias(tl.load(x_ptr))       # scalar load -> tt.call -> scalar f32
        tl.store(o_ptr + offs, x + b + (pid + npr).to(tl.float32))

    return ops_func_pid_call_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    o = torch.zeros(N, dtype=torch.float32)
    bias = x[0] * 3.0 + 1.0
    for pid in range(GRID):
        s = slice(pid * BLOCK, (pid + 1) * BLOCK)
        o[s] = x[s] + bias + float(pid + GRID)
    return {"o": o}


SPEC = KernelSpec(
    name="ops_func_pid_call_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(GRID,),
    reference=reference,
    make_inputs=make_inputs,
)
