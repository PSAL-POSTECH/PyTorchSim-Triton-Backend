"""The tensor-of-!tt.ptr family: the four converters that only exist because a
pointer op reached pass 4 still operating on a TENSOR.

Split out of the old tests/cases/ops.py, and it keeps its own file because
its constraint is real and fragile -- see WHY THE INT_TO_PTR MUST BE A TENSOR.
Every other file in the ops_* split has ordinary affine addresses that pass 1
rewrites into tts.make_tptr; this one deliberately builds pointers that pass 3
never even looks at.

COVERS -- 6 branch ids.  Line numbers are the triton-shared WORKING TREE,
source root /workspace/triton_shared/.

  opt-tensor-ptr-to-linalg-DIVERGES   include/triton-shared/Conversion/
                                      TritonArithToLinalg/Passes.td:24
      Option<"tensorPtrToLinalg", default=FALSE> -- and stage 3 passes
      `--triton-arith-to-linalg=tensor-ptr-to-linalg=true` (pipeline.py's
      TTSHARED_PIPELINE).  That is a genuine DEFAULT MISMATCH: standalone
      `triton-shared-opt --triton-arith-to-linalg` behaves differently from the
      pipeline this harness runs.  When true,
      populateTritonTensorPtrConversionPatterns is registered and tt.load /
      tt.store / tt.int_to_ptr / tt.ptr_to_int become dynamically illegal for
      ShapedType operand 0, while tt.bitcast's legality FLIPS -- a ptr-typed
      tensor bitcast becomes illegal instead of legal
      (TritonArithToLinalgPass.cpp:163-183).  A .td option: gdb cannot break on
      it, so it is verified by the four generics below existing at all.  This
      file is the only witness for it in the whole set.

  opt-addptr-to-linalg     Passes.td:20
      Option<"addptrToLinalg", default=true>: registers AddPtrConverter and
      marks tt.addptr dynamically illegal ONLY for ShapedType results
      (TritonArithToLinalgPass.cpp:157-161); a scalar tt.addptr stays legal.
      Also .td, also verified by IR.

  ptr-load-generic         lib/Conversion/TritonArithToLinalg/
                           TritonArithToLinalg.cpp:42
      TensorOpConverter<triton::LoadOp> (ConversionPatterns.hpp:2188).

  ptr-cast-generic         TritonArithToLinalg.cpp:43
      TensorOpConverter<IntToPtrOp / PtrToIntOp / BitcastOp>.  UNIT CASE: all
      three emit byte-identical structure from the same template body
      (ConversionPatterns.hpp:2180-2217) and differ only in the name of the
      scalar op cloned into the region.  This file has all three anyway, because
      they are what the address chain is made of.

  ptr-store-generic        TritonArithToLinalg.cpp:42
      StorePtrToLinalgConverter (ConversionPatterns.hpp:2228).

  addptr-generic           TritonArithToLinalg.cpp:57
      AddPtrConverter (ConversionPatterns.hpp:2148).

WHAT PROVES IT FIRED   work/ops_tensor_of_ptrs_kernel/triton-shared-detail/

  02-unstructured.mlir is the load-bearing one: pass 3 left every pointer op
  RAW.  There is not one tts.make_tptr, tts.gather or memref op in the whole
  module -- it is still a `tt.func` full of `tt.*`, which no other file in this
  split produces.  The tt.addptr / tt.load / tt.store / tt.ptr_to_int /
  tt.bitcast below walked through pass 1 and pass 3 untouched into pass 4:

      %7  = tt.int_to_ptr %6 : tensor<128xi64> -> tensor<128x!tt.ptr<f32>>
      %8  = tt.load %7 : tensor<128x!tt.ptr<f32>>
      %9  = tt.ptr_to_int %7 : tensor<128x!tt.ptr<f32>> -> tensor<128xi64>
      %10 = tt.bitcast %7 : tensor<128x!tt.ptr<f32>> -> tensor<128x!tt.ptr<i32>>
      %11 = tt.load %10 : tensor<128x!tt.ptr<i32>>
      %18 = tt.int_to_ptr %17 : tensor<128xi64> -> tensor<128x!tt.ptr<f32>>
      %19 = tt.addptr %18, %12 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %19, %25 : tensor<128x!tt.ptr<f32>>

  03-arith-linalg.mlir turns each into a linalg.generic whose BODY is the scalar
  version of the same op -- that is the whole family.  ptr-load-generic (%21 and
  %27); note `outs` is a fresh tensor.empty and NOT an operand reuse:

      %20 = tensor.empty() : tensor<128xf32>
      %21 = linalg.generic {indexing_maps = [#map, #map],
                            iterator_types = ["parallel"]}
              ins(%19 : tensor<128x!tt.ptr<f32>>) outs(%20 : tensor<128xf32>) {
      ^bb0(%in: !tt.ptr<f32>, %out: f32):
        %47 = tt.load %in : !tt.ptr<f32>
        linalg.yield %47 : f32
      } -> tensor<128xf32>

  ptr-cast-generic, four times over three op names, and the census's "UNIT CASE"
  claim is literally true -- the generics are byte-identical apart from the
  cloned op and the result element type:

      %19  ^bb0(%in: i64, %out: !tt.ptr<f32>):
             %47 = tt.int_to_ptr %in : i64 -> !tt.ptr<f32>
      %23  ^bb0(%in: !tt.ptr<f32>, %out: i64):
             %47 = tt.ptr_to_int %in : !tt.ptr<f32> -> i64
      %25  ^bb0(%in: !tt.ptr<f32>, %out: !tt.ptr<i32>):
             %47 = tt.bitcast %in : !tt.ptr<f32> -> !tt.ptr<i32>
      %37  ^bb0(%in: i64, %out: !tt.ptr<f32>):     (the store's base)
             %47 = tt.int_to_ptr %in : i64 -> !tt.ptr<f32>

  addptr-generic is the one with a DIFFERENT shape, and that difference is the
  branch: exactly THREE hardcoded identity maps, and outs = the POINTER operand
  (%37) rather than a tensor.empty:

      %38 = linalg.generic {indexing_maps = [#map, #map, #map],
                            iterator_types = ["parallel"]}
              ins(%37, %28 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>)
              outs(%37 : tensor<128x!tt.ptr<f32>>) {
      ^bb0(%in: !tt.ptr<f32>, %in_1: i32, %out: !tt.ptr<f32>):
        %47 = tt.addptr %in, %in_1 : !tt.ptr<f32>, i32
        linalg.yield %47 : !tt.ptr<f32>
      } -> tensor<128x!tt.ptr<f32>>

  ptr-store-generic is the last op in the function and the only linalg.generic
  in the whole set with NO results and no `outs` clause -- the converter pushes
  nothing into `outputs`, so the yield takes no operands:

      linalg.generic {indexing_maps = [#map, #map],
                      iterator_types = ["parallel"]}
        ins(%38, %46 : tensor<128x!tt.ptr<f32>>, tensor<128xf32>) {
      ^bb0(%in: !tt.ptr<f32>, %in_1: f32):
        tt.store %in, %in_1 : !tt.ptr<f32>
        linalg.yield
      }

WHY THE int_to_ptr MUST BE A TENSOR  (do not "simplify" this kernel)
  TritonToUnstructuredPass.cpp:258 SKIPS RankedTensorType results when seeding
  its offset map.  A tensor tt.int_to_ptr is therefore never seeded, so pass 3
  never visits it or anything downstream of it, and the whole chain survives raw
  into pass 4 -- which is the only way these converters ever run.  Build the
  same pointers from a scalar int_to_ptr plus a splat and pass 3 seeds them, the
  ops are rewritten into tts/memref, and this file covers nothing.

  The lo/hi split on the STORE address is the second half of that: it makes the
  per-element address arithmetic genuinely per-element, so the canonicalizer
  cannot fold `int_to_ptr(splat(scalar) + i*4)` back into
  `splat(int_to_ptr(scalar))` and hand the address back to PtrAnalysis.

HOW THE BACKEND TAKES IT -- it runs end to end, and the gap it used to name
  is closed.  This file used to say it stopped at run.py stage 2 on a
  `tptr.type_offset` the backend had no dialect for; that has not been true for
  some time.  What it stopped on next was bufferization, which is the honest
  wall: `cannot bufferize value of type 'tensor<128x!tt.ptr<f32>>'` -- there is
  no memref whose element type is a pointer.

  convert_unsupported_pointers DISSOLVES the pointer element type.  A !tt.ptr element
  is a 64-bit address and nothing else, so the three cast generics above are the
  identity on that address and vanish, the addptr generic becomes
  `p + extsi(i)*4` on i64, and the load/store generics become the unstructured
  tts.gather / tts.scatter this backend already lowers -- measured against a
  real reference at max_abs_err = 0.  See _convert_unsupported_pointers.

  WHAT THAT DOES NOT CHANGE is anything this file is coverage FOR.  The six
  branch ids above are all upstream, and they are measured on 03-ttshared.mlir,
  which still contains exactly the generics quoted here: the backend reads that
  file, it does not produce it.  Keep building the pointers this way.

ARITHMETIC
  u[i] = 2*x[i], written through a hand-built address.  `keep` is
  (ptr_bits ^ reinterpreted_float_bits) as f32 times 0.0 -- it exists only to
  keep the tt.ptr_to_int and the bitcast-to-int32 pointer live, and sitofp of
  any integer is finite, so it contributes exactly 0.0.

SCOPE
  IR-census kernel.  The four code ids were confirmed by gdb line hits
  (the branch census); the two `opt-` ids are TableGen options with no
  line to break on, so the four generics quoted above ARE their measurement --
  those patterns exist in the pattern set only when the options are true.

POSITIVE KERNEL, end to end.

The tensor-of-!tt.ptr converters (load/store/cast/addptr generics) and the two pass options that register them. The backend dissolves the pointer element type into i64 and gathers -- see the module docstring.
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
    def ops_tensor_of_ptrs_kernel(x_ptr, u_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)

        xb = tl.cast(x_ptr, tl.int64)                 # SCALAR tt.ptr_to_int
        ub = tl.cast(u_ptr, tl.int64)
        xaddr = xb + offs.to(tl.int64) * 4
        # TENSOR tt.int_to_ptr -- the one construct pass 3 refuses to seed.
        xp = tl.cast(xaddr, tl.pointer_type(tl.float32))
        gg = tl.load(xp)                              # tt.load, tensor of ptrs
        xi = tl.cast(xp, tl.int64)                    # TENSOR tt.ptr_to_int
        xq = tl.cast(xp, tl.pointer_type(tl.int32), bitcast=True)  # tt.bitcast
        gi = tl.load(xq)

        # lo/hi split: per-element address arithmetic the canonicalizer cannot
        # fold back into splat(int_to_ptr(scalar)).  lo + hi_off == offs.
        lo = offs & 63
        hi_off = offs - lo
        uaddr = ub + hi_off.to(tl.int64) * 4
        up = tl.cast(uaddr, tl.pointer_type(tl.float32))
        ups = up + lo                                 # tt.addptr, tensor of ptrs

        keep = (xi.to(tl.int32) ^ gi).to(tl.float32) * 0.0
        tl.store(ups, gg * 2.0 + keep)                # tt.store, tensor of ptrs

    return ops_tensor_of_ptrs_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    # keep == 0 exactly: sitofp of any integer is finite, and finite * 0.0 == 0.
    return {"u": inputs["x"] * 2.0}


SPEC = KernelSpec(
    name="ops_tensor_of_ptrs_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "u_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("u", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
