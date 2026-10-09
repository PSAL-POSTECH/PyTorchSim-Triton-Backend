"""NEGATIVE: a masked load of a TENSOR OF POINTERS whose `other` is a POINTER.

Split out of the old tests/cases/ptr_reject.py (one case per file).
This kernel is expected to fail; the failure is the observable.

COVERS

  tts-load-other-is-pointer
      include/triton-shared/Dialect/TritonStructured/IR/
        TritonStructuredDialect.td:316
      TTS_LoadOp's `$other` argument is an AnyTypeOf that explicitly admits
      TT_Ptr, i.e. a masked load of a tensor of pointers whose fill value is
      itself a pointer. Nothing else in the kernel set builds that operand.

  This is a TABLEGEN (ODS) branch, not C++, so the branch census
  cannot measure it -- gdb has nothing to break on and the tool lists it
  under "not measurable (.td, no code)". It is verified by READING the IR,
  which is what the census's `observable` field asks for anyway.

WHAT PROVES IT FIRED  (quoted from
  work/ptr_load_other_is_pointer_kernel/triton-shared-detail/01-structured.mlir)

      %3 = tts.make_tptr %arg1 to sizes: [8], strides: [1], offsets: [0],
             shape: [0], order: []
             : <!tt.ptr<f32>> to tensor<8x!tt.ptr<!tt.ptr<f32>>>
      %6 = "tts.load"(%3, %arg0)
             <{operandSegmentSizes = array<i32: 1, 0, 1>,
               static_mask_dims = array<i64: 4>}>
             : (tensor<8x!tt.ptr<!tt.ptr<f32>>>, !tt.ptr<f32>)
             -> tensor<8x!tt.ptr<f32>>

  Read the operand list: the tts.load takes a tensor of POINTERS TO POINTERS
  and a bare `!tt.ptr<f32>` as its third operand, and returns
  `tensor<8x!tt.ptr<f32>>`.  That third operand is `$other`, and its type is
  the TT_Ptr alternative of the AnyTypeOf -- nothing else in the kernel set
  produces it.  `operandSegmentSizes = array<i32: 1, 0, 1>` confirms which
  slot it is: 1 pointer, 0 mask operands (the mask is the static
  `array<i64: 4>`), 1 other.

WHY IT IS INTERESTING
  Everywhere else in the linalg lowering a load's `other` is a numeric fill value
  and the loaded element type is a scalar type. Here the buffer is `**fp32`,
  so the load's result element type is `!tt.ptr<f32>` and the fill has to be a
  pointer too. It is the only construct in the whole set that instantiates the
  TT_Ptr alternative of that AnyTypeOf.

WHY IT FAILS, AND WHY THAT IS CORRECT
  There is no memref whose element type is a pointer.  Pass 1 builds the
  tts.load happily, --triton-to-unstructured passes it through, and
  --triton-ptr-to-memref then writes the `**fp32` argument as
  `memref<*x!tt.ptr<f32>>` anyway -- a type MLIR rejects.

  STAGE 3 EMITS A MODULE IT CANNOT READ BACK, and that is the observable.
  triton-shared-opt exits 0 and writes 07-regularized.mlir; handing that same file
  straight back to it fails:

      07-regularized.mlir:2:60: error: invalid memref element type
        "func.func"() <{function_type = (memref<*xf32>,
                                         memref<*x!tt.ptr<f32>>, ...)

  So stage 4 refuses on the parse, not on anything of ours -- the MLIR bindings
  reject the signature before a single pass of ours runs.  This used to be a
  SIGABRT inside --structured-to-memref, which asserted while BUILDING the
  type; that pass is deleted (rule 21) and the invalid type now survives to a
  file.  The defect is the same one, one pass later and quieter, and it is
  open on the linalg lowering side.

HARNESS NOTE ON THE ARGUMENT TYPES
  `qq` is declared int64 in `args` while the triton signature says `**fp32`:
  the harness has no pointer dtype, and the buffer only has to type-check
  because the file never produces an executable. Same trick the old
  ptr_reject.py used.  `--to triton-compile` SUCCEEDS on this kernel now; it is
  stage 4 that stops, so do not read a green stage 3 as the type being legal.

reference() is an unreachable stub: the value written into qq[4:8] is the raw
ADDRESS of in_ptr, which torch cannot know.

NEGATIVE, .td branch: tts-load-other-is-pointer. Verified by reading 01-structured.mlir -- gdb cannot break on TableGen.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

N = 1024
R = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_load_other_is_pointer_kernel(in_ptr, qq_ptr, R: tl.constexpr):
        r8 = tl.arange(0, R)

        # tts.load whose `other` is a POINTER (TTS_LoadOp's TT_Ptr alternative)
        pv = tl.load(qq_ptr + r8, mask=r8 < 4, other=in_ptr)
        tl.store(qq_ptr + r8, pv)

    return ptr_load_other_is_pointer_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in": torch.randn(N, generator=g, dtype=torch.float32),
            "qq": torch.zeros(R, dtype=torch.int64)}


def reference(inputs):
    """UNREACHABLE STUB -- the module never lowers past pass 1.

    qq[0:4] would keep their loaded values and qq[4:8] would receive the raw
    address of in_ptr, which no torch expression can produce.
    """
    return {"qq": inputs["qq"]}


SPEC = KernelSpec(
    name="ptr_load_other_is_pointer_kernel",
    kernel=kernel,
    signature={"in_ptr": "*fp32", "qq_ptr": "**fp32", "R": "constexpr"},
    constexprs={"R": R},
    args=[Arg("in", IN, "float32", (N,)),
          Arg("qq", INOUT, "int64", (R,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="refuse",
    expect_why="a masked load whose `other` is a tensor of pointers",
)
