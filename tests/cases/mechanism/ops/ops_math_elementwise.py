"""The scalar-math converters of pass 4: min/max, clamp, precise, extern, and
the two one-liners (umulhi, non-pointer bitcast).

Split out of the old tests/cases/ops.py.  One file per converter FAMILY,
and this is the family whose every member has the same shape: load a 1-D
vector, apply one triton op, keep the result live.  Nothing here needs setup --
no second function, no frontend option, no pointer games -- so splitting it
further would produce six four-line kernels that read identically and differ
only in which line of TritonArithToLinalg.cpp registered the pattern.

COVERS -- 12 branch ids.  Line numbers are the triton-shared WORKING TREE,
source root /workspace/triton_shared/.

  A. MinMaxConverter, the GREEDY pre-pass
  ---------------------------------------
  canon-minmax-cmpf-select      lib/Conversion/TritonArithToLinalg/
                                TritonArithToLinalg.cpp:36
      MinMaxConverter<arith::CmpFOp> path 1 (ConversionPatterns.hpp:1902): the
      cmpf's single user is an arith.select whose true/false values are the
      cmp's own lhs/rhs.  useNaNOps is hardcoded true here, so the result is
      always maximumf/minimumf, never maxnumf/minnumf.

  canon-minmax-cmpi-select      TritonArithToLinalg.cpp:36
      MinMaxConverter<arith::CmpIOp>, same path, integer overload
      (rewriteOpWithMinMax at ConversionPatterns.hpp:2024).  cmpi has ONLY this
      arm: the or-path calls foldCmpSelectToMinMax, which bails on
      !isa<FloatType>(sel.getType()) (ConversionPatterns.hpp:1946).

  canon-minmax-cmpf-or-select   TritonArithToLinalg.cpp:36
      MinMaxConverter<arith::CmpFOp> path 2: findSelectThroughOr walks
      cmpf -> arith.ori -> arith.select and foldCmpSelectToMinMax matches
      (ogt(a,b) || une(a,a)) ? a : b   (ConversionPatterns.hpp:1915, 1943-2009).

  B. ClampConverter, both propagate_nan policies
  -----------------------------------------------
  clamp-propagate-nan   TritonArithToLinalg.cpp:75   ConversionPatterns.hpp:976
      PropagateNan::ALL  -> arith.maximumf then arith.minimumf (NaN-aware).
  clamp-numeric         TritonArithToLinalg.cpp:75   ConversionPatterns.hpp:979
      PropagateNan::NONE -> arith.maxnumf  then arith.minnumf  (numeric).

  C. the two IEEE-precision converters
  -------------------------------------
  precise-sqrt   TritonArithToLinalg.cpp:69   tt.precise_sqrt -> math.sqrt
  precise-div    TritonArithToLinalg.cpp:70   tt.precise_divf -> arith.divf

  D. tt.extern_elementwise, both arities
  ---------------------------------------
  extern-elementwise-unary    TritonArithToLinalg.cpp:83
      ExternElementwiseUnaryOpConverter: pure, getSrcs().size() == 1, and an
      EXACT match against a ~50-entry table of __nv_* names
      (ConversionPatterns.hpp:2342-2389).
  extern-elementwise-binary   TritonArithToLinalg.cpp:83
      ExternElementwiseBinaryOpConverter: pure, getSrcs().size() == 2, symbol in
      {__nv_atan2f, __nv_atan2, __nv_powf, __nv_pow}
      (ConversionPatterns.hpp:2307-2318).

  E. the two unconditional one-liners
  ------------------------------------
  mulhiui-extended   TritonArithToLinalg.cpp:68
      MulHiUIOpConverter: tt.mulhiui -> arith.mului_extended, replaced with the
      `high` result ONLY; the `low` result is dropped and dies.
  bitcast-arith      TritonArithToLinalg.cpp:66
      BitcastConverter, the NON-pointer half of its guard -- it returns failure
      immediately if triton::isPtrTypeLike(op.getType())
      (ConversionPatterns.hpp:866).  The pointer half is a different branch
      (ptr-cast-generic) and lives in ops_tensor_of_ptrs.py.

  F. and the pattern that cleans up after all of them
  ----------------------------------------------------
  elementwise-to-linalg   TritonArithToLinalg.cpp:101
      linalg::populateElementwiseToLinalgConversionPatterns, added LAST so it is
      tried last.  An arith/math op is illegal iff ALL its operand types are
      RankedTensorType (TritonArithToLinalgPass.cpp:145-150), so every tensor
      arith./math. op the converters above just produced gets wrapped into a
      linalg.generic.  The claim is parked in this file because EVERY
      linalg.generic here comes from that pattern: there is no broadcast (which
      would carry a broadcastDims attribute), no addptr generic and no
      tensor-of-pointer generic to confuse it with.

WHAT PROVES IT FIRED   work/ops_math_elementwise_kernel/triton-shared-detail/

  Pass 1 has not touched any of it -- the greedy pre-pass belongs to pass 4 --
  so 01-structured.mlir:21-46 still shows every input shape raw:

      %13 = arith.cmpf ogt, %4, %8 : tensor<128xf32>
      %14 = arith.select %13, %4, %8 : tensor<128xi1>, tensor<128xf32>
      %16 = arith.cmpi sgt, %12, %15 : tensor<128xi32>
      %17 = arith.select %16, %12, %15 : tensor<128xi1>, tensor<128xi32>
      %18 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
        %43 = arith.cmpf ogt, %arg6, %arg7 : f32
        %44 = arith.cmpf une, %arg6, %arg6 : f32
        %45 = arith.ori %43, %44 : i1
        %46 = arith.select %45, %arg6, %arg7 : f32
      %19 = tt.clampf %14, %cst_2, %cst_1, propagateNan = all  : tensor<128xf32>
      %20 = tt.clampf %14, %cst_2, %cst_1, propagateNan = none : tensor<128xf32>
      %22 = tt.precise_sqrt %21 : tensor<128xf32>
      %25 = tt.precise_divf %22, %24 : tensor<128xf32>
      %26 = tt.extern_elementwise %25 {... symbol = "__nv_expf"}
      %27 = tt.extern_elementwise %26, %8 {... symbol = "__nv_atan2f"}
      %28 = tt.mulhiui %12, %12 : tensor<128xi32>
      %29 = tt.bitcast %27 : tensor<128xf32> -> tensor<128xi32>
      %31 = tt.bitcast %30 : tensor<128xi32> -> tensor<128xf32>

  03-arith-linalg.mlir has ZERO arith.select, arith.cmpf, arith.cmpi and
  arith.ori left, and zero tt.* OPERATIONS -- `!tt.ptr` survives only as a type
  in the signature.  Each cmp+select became exactly one min/max op, one per arm:

      %16 = ... { %43 = arith.maximumf %in, %in_3 : f32 }   A path 1 (cmpf+select)
      %18 = ... { %43 = arith.maxsi    %in, %in_3 : i32 }   A cmpi+select
      %reduced = linalg.reduce ins(%11) outs(%inserted) dimensions = [0]
        (%in: f32, %init: f32) { %43 = arith.maximumf %in, %init : f32 }
                                                           A path 2, the or-path

  The or-path is load-bearing for that linalg.reduce existing at all: with the
  ori/cmpf/select still in place, ReduceConverter would see four ops in the
  tt.reduce body instead of one supported reduction op and would refuse it.

  The two clamps become four adjacent linalg.generics (%20..%23), identical
  except for the spelling -- which is the entire branch:

      %20  { arith.maximumf %in, %in_3 }   \\ clamp-propagate-nan (ALL)
      %21  { arith.minimumf %in, %in_3 }   /
      %22  { arith.maxnumf  %in, %in_3 }   \\ clamp-numeric (NONE)
      %23  { arith.minnumf  %in, %in_3 }   /

  and the rest, one generic each:

      %25  { %43 = math.sqrt  %in : f32 }              precise-sqrt
      %28  { %43 = arith.divf %in, %in_3 : f32 }       precise-div
      %29  { %43 = math.exp   %in : f32 }              extern unary  __nv_expf
      %30  { %43 = math.atan2 %in, %in_3 : f32 }       extern binary __nv_atan2f
      %33  { %43 = arith.bitcast %in : f32 to i32 }    bitcast-arith
      %36  { %43 = arith.bitcast %in : i32 to f32 }    bitcast-arith, back again

  mulhiui-extended is the one with a visibly odd shape -- a TWO-result
  linalg.generic, because arith.mului_extended has two results:

      %31:2 = linalg.generic ... ins(%15, %15) outs(%15, %15) {
      ^bb0(%in: i32, %in_3: i32, %out: i32, %out_4: i32):
        %low, %high = arith.mului_extended %in, %in_3 : i32
        linalg.yield %low, %high : i32, i32
      } -> (tensor<128xi32>, tensor<128xi32>)

  and the store reads `%31#1` -- the HIGH half.  `%31#0` (low) has no user at
  all: that is the census's "replaced with the high result only, the low result
  is dropped" made visible.

  Every scalar op above is the BODY of a linalg.generic with all-identity
  indexing maps, all-parallel iterators and NO broadcastDims attribute -- that
  wrapping is elementwise-to-linalg, and it is the only source of
  linalg.generic in this module.

WHY THE OR-PATH NEEDS A tl.reduce
  foldCmpSelectToMinMax opens with `if (!isa<FloatType>(sel.getType())) return
  failure();` and a tensor select has type tensor<128xf32>, which is not a
  FloatType.  The or-path can therefore fire only on a SCALAR select, and the
  only place triton emits scalar float arithmetic is inside a tt.reduce /
  tt.scan region body.  `tl.reduce(x, 0, _nanmax)` is the smallest thing that
  produces one.  It is scaffolding: the reduce-vector branch it also triggers
  is claimed by ops_reduce.py, not here.

ARITHMETIC
  Deliberately trivial and exact.  |atan2(exp(|clamp(max(x,y))| / (clamp^2+1)),
  y)| + max(x), plus an integer max and umulhi(n, n) on the side.  The bitcast
  pair is the classic fabs-via-clearing-the-sign-bit idiom, which is also why
  there are two arith.bitcast ops in opposite directions.

WHAT THE BLOCK USED TO BE, AND WHAT IT IS NOW
  This file was blocked on libdevice.atan2: math.atan2 went through
  convert-math-to-llvm to an intrinsic RISC-V has no vector form for, so llc
  scalarised it into one `call atan2f` per element -- and there is one scalar
  register file for every lane.  convert_unsupported_ops now rewrites
  atan2 into math.atan plus a quadrant select before it ever reaches LLVM, and
  06-kernel.s for this kernel has no `call` in it at all.  THAT BLOCK HAS
  LIFTED; ops_math_derived.py is its positive replacement.

  WHAT BLOCKS IT NOW IS A SPILL, and the first answer here was wrong.  It said
  tl.umulhi, because replacing the umulhi alone with `n * 2` takes this kernel
  to ALL STAGES PASSED.  It does -- but not for the reason given.  The
  two-result generic WAS mishandled (vectorize stored one value where two
  belonged; see ops_umulhi.py, which pins that on its own and passes), and
  fixing it did NOT unblock this file.  What the substitution actually changed
  was register pressure.

      vs8r.v v8, (t4)     # vscale x 64-byte Folded Spill
      VI_ST_WHOLE ERROR: Accessing invalid spad address: 0x3ffffff490

  llc spills an LMUL=8 register group to an sp-relative slot, and the stack is
  not in the banked scratchpad window, so spike refuses the whole-register
  store.  There is no legal place to spill a vector here, which makes any spill
  fatal -- see the rule book under vectorize for why this is not a
  live-value count.

SCOPE
  IR-census kernel.  Verified two ways: the branch census put a gdb
  breakpoint on every census line and all 12 ids above came back with non-zero
  hits, and the IR quoted above was read out of tools/lower_steps.py's output.
  Like every file under tests/cases/ it does NOT reach run.py's numeric
  stages -- stage 3 (pytorchsim_triton_compiler/lowering/passes/) only handles the narrow shape
  tests/cases/smoke/add.py has, and tests/cases/harness/gap_fill.py stops there too.  The
  torch reference below is kept exact so the file becomes runnable if that
  changes.

POSITIVE KERNEL: every triton-shared pass succeeds.

AND THE REGISTER-FILE BUDGET, which it is the only witness for. Its atan2 loop keeps ~12 vectors live; at the old element count (64 f32, LMUL 8) the file holds four, so llc emitted vs8r.v to the stack and spike refused the 256-byte whole-group store (VI_ST_WHOLE ERROR). legalize now fits the file too and picks 32. Do not widen the body without re-checking the .s for `Folded Spill`. Pass-4 scalar-math converter family: MinMax (3 arms), Clamp (2), PreciseSqrt/PreciseDiv, ExternElementwise (2 arities), MulHiUI, the non-pointer BitcastConverter arm, and the elementwise-to-linalg wrapping that follows all of them. POSITIVE.
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
    from triton.language.extra import libdevice
    # triton resolves names from the DEFINING function's globals, and these are
    # locals of kernel(); publish them so the @triton.jit body can see them.
    globals()["libdevice"] = libdevice

    @triton.jit
    def _nanmax(a, b):
        # scalar cmpf ogt + cmpf une + ori + select -- the ONLY shape that
        # reaches foldCmpSelectToMinMax's or-path.
        return tl.where((a > b) | (a != a), a, b)

    globals()["_nanmax"] = _nanmax

    @triton.jit
    def ops_math_elementwise_kernel(x_ptr, y_ptr, i_ptr, o_ptr, p_ptr, q_ptr,
                                    BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r)
        y = tl.load(y_ptr + r)
        n = tl.load(i_ptr + r)

        # A. MinMaxConverter -- three arms, three shapes of cmp+select.
        mf = tl.where(x > y, x, y)              # cmpf ogt + select -> maximumf
        mi = tl.where(n > n * 2, n, n * 2)      # cmpi sgt + select -> maxsi
        nm = tl.reduce(x, 0, _nanmax)           # scalar cmpf|ori|select -> maximumf

        # B. ClampConverter -- same input, both policies, so the ONLY difference
        #    in the emitted IR is the arith spelling.  They cannot CSE into each
        #    other precisely because the propagate_nan attribute differs.
        ca = tl.clamp(mf, -0.5, 0.5, propagate_nan=tl.PropagateNan.ALL)
        cn = tl.clamp(mf, -0.5, 0.5)            # default: PropagateNan.NONE

        # C. precise math.  ca*ca instead of tl.abs so no other math.* op in the
        #    module could be mistaken for the math.sqrt witness.
        sq = tl.sqrt_rn(ca * ca)                # tt.precise_sqrt
        dv = tl.div_rn(sq, cn * cn + 1.0)       # tt.precise_divf

        # D. extern elementwise.  libdevice.* is what emits tt.extern_elementwise
        #    at all -- plain tl.exp lowers to math.exp in the FRONTEND and would
        #    never produce an extern op for the converter to match.
        ee = libdevice.exp(dv)                  # __nv_expf   -- 1 src
        a2 = libdevice.atan2(ee, y)             # __nv_atan2f -- 2 srcs

        # E. umulhi, and a non-pointer bitcast ROUND TRIP.
        hi = tl.umulhi(n, n)                              # tt.mulhiui
        bits = a2.to(tl.int32, bitcast=True)              # tt.bitcast f32->i32
        absf = (bits & 0x7FFFFFFF).to(tl.float32, bitcast=True)   # i32->f32

        tl.store(o_ptr + r, absf + nm)
        tl.store(p_ptr + r, mi)
        tl.store(q_ptr + r, hi)

    return ops_math_elementwise_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "y": torch.randn(N, generator=g, dtype=torch.float32),
            # Big and non-negative: the square genuinely overflows 32 bits (so
            # `hi` is non-zero) while staying representable in an i32 argument
            # and unambiguous under umulhi's UNSIGNED reading.  Signedness also
            # makes `n > 2n` false for every element, which is fine -- the
            # branch is about the emitted maxsi, not about which side wins.
            "i": torch.randint(1 << 20, 1 << 30, (N,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    x, y, n = inputs["x"], inputs["y"], inputs["i"]
    mf = torch.maximum(x, y)
    c = mf.clamp(-0.5, 0.5)          # ALL and NONE agree: randn has no NaNs
    sq = torch.sqrt(c * c)
    dv = sq / (c * c + 1.0)
    a2 = torch.atan2(torch.exp(dv), y)
    wide = n.to(torch.int64) * n.to(torch.int64)
    return {"o": a2.abs() + x.max(),
            "p": torch.maximum(n, n * 2),
            "q": (wide >> 32).to(torch.int32)}


SPEC = KernelSpec(
    name="ops_math_elementwise_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "i_ptr": "*i32",
               "o_ptr": "*fp32", "p_ptr": "*i32", "q_ptr": "*i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", IN, "float32", (N,)),
          Arg("i", IN, "int32", (N,)),
          Arg("o", OUT, "float32", (N,)), Arg("p", OUT, "int32", (N,)),
          Arg("q", OUT, "int32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
