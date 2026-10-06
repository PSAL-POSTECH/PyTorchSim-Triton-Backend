"""bitcast of a SCALAR pointer -- the tt.bitcast op itself becomes the new base.

Split out of the old tests/cases/ptr_reject.py (one case per file).

COVERS  (all line numbers are the triton-shared WORKING TREE, source root
         /workspace/triton_shared/)

  visitoperand-ptr-bitcast   lib/AnalysisStructured/PtrAnalysis.cpp:1324
      A scalar !tt.ptr operand whose defining op is a triton::BitcastOp, so
      visitOperand dispatches to visitOperandBitcast instead of failing.

  bitcast-scalar-newsource   lib/AnalysisStructured/PtrAnalysis.cpp:1281
      Inside visitOperandBitcast: the RESULT is not a ShapedType (this is a
      scalar pointer, not a tensor of pointers), so the analysis stops
      recursing and does `state.source = op.getResult()` -- the bitcast
      becomes the base rather than being looked through.

WHAT PROVES IT FIRED  (quoted from
  work/ptr_bitcast_scalar_base_kernel/triton-shared-detail/01-structured.mlir)

      %1 = tt.bitcast %arg0 : !tt.ptr<f32> -> !tt.ptr<i32>
      %6 = tts.make_tptr %1 to sizes: [128], strides: [1], offsets: [0],
             shape: [0], order: [] : <i32> to tensor<128x!tt.ptr<i32>>
      %8 = "tts.load"(%6) ... -> tensor<128xi32>

  The make_tptr base is `%1`, the BITCAST -- not `%arg0`.  That is the whole
  difference between :1281 (stop, use the bitcast) and bitcast-shaped-recurse
  (walk through to the source).  Compare the store on the same line, whose
  base really is a function argument: `tts.make_tptr %arg1 to ...`.
  Analysis SUCCEEDS here; this is a rejection only in the sense that the walk
  refuses to look past the bitcast.

WHY IT IS INTERESTING
  The old ptr_reject.py docstring flagged this as the suspected killer of the
  whole module and it is not: on its own the scalar bitcast lowers all the way
  to memref.  05-unstruct-memref.mlir carries the bitcast through to the
  memref world unchanged:

      %1 = tt.bitcast %arg0 : !tt.ptr<f32> -> !tt.ptr<i32>
      %2 = builtin.unrealized_conversion_cast %1 : !tt.ptr<i32> to memref<*xi32>
      %reinterpret_cast_0 = memref.reinterpret_cast %2 to offset: [0],
                              sizes: [128], strides: [1]
        : memref<*xi32> to memref<128xi32, strided<[1]>>

  i.e. the pointer bitcast survives as a memref element-type change and
  nothing rejects it.

WHAT THIS BACKEND DOES WITH IT
  It runs, end to end, and the answer is exact.  convert_accesses_to_transfer's
  base_memref walks THROUGH the scalar bitcast to the
  builtin.unrealized_conversion_cast under it and returns %arg0 -- a pointer
  bitcast changes nothing about the address -- and the transfer takes its
  element type from the TILE, so an i32 tile gets an i32 scratchpad even though
  the argument is declared memref<*xf32>.  The two widths must agree; they do
  here (4 bytes each) and a mismatch is refused by name.

ARITHMETIC
  Deliberately none: reinterpret the f32 input's bits as i32 and store them.
  reference() is the exact torch equivalent, `x.view(torch.int32)`.

visitoperand-ptr-bitcast + bitcast-scalar-newsource. The bitcast is RETAINED as the tts.make_tptr base instead of being looked through, so base_memref has to walk one op further than usual.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_bitcast_scalar_base_kernel(in_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)

        # bitcast of a scalar pointer -> the bitcast becomes the new base
        pb = tl.cast(in_ptr, tl.pointer_type(tl.int32))
        tl.store(o_ptr + r, tl.load(pb + r))

    return ptr_bitcast_scalar_base_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    # the same bytes, read as i32 -- exactly what tt.bitcast on the base means
    return {"o": inputs["x"][:BLOCK].contiguous().view(torch.int32)}


SPEC = KernelSpec(
    name="ptr_bitcast_scalar_base_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "o_ptr": "*i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)),
          Arg("o", OUT, "int32", (BLOCK,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
