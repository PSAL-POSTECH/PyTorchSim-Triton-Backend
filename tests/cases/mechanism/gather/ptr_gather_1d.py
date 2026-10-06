"""The plain 1-D gather x[i] -- PtrAnalysis's whole gather/scatter rebuild in
its simplest possible form.

This is the base case the other 14 ptr_gather_*.py files are variations of. The
address is `x_ptr + i` where `i` came out of memory, so visitOperand hits an op
it cannot model, and instead of failing it REBUILDS the state with the loaded
tensor itself as the offset of the one non-contiguous dimension.

COVERS (5)
  visitoperand-default-rebuild-unsupported   PtrAnalysis.cpp:1391
      the TypeSwitch .Default arm: unsupported defining op AND
      enableMakeGatherScatterTensorPtr -> state.rebuildAsUnsupportedOp(operand)
      instead of returning failure.
  rebuild-unsupported-success                PtrAnalysis.cpp:232
      the per-dim loop inside it: size == 1 -> offset 0 / stride 0; size > 1 ->
      offset = the operand Value / stride 1, shape 0. Exactly one dim may have
      size > 1, which is what makes the result representable.
  addptr-gs-success                          PtrAnalysis.cpp:1434
      rewriteAddptrOp: not structured, gather/scatter enabled, the second
      (unstructured) analysis pass and the merge both succeeded -> knownPtrs is
      overwritten with the MERGED state.
  gs-op-emitted                              PtrAnalysis.cpp:704
      createTTSMakeGatherScatterTensorPtrOp's tail, the 6-operand builder with
      no mask.
  gs-offset-1d-no-collapse                   PtrAnalysis.cpp:686
      the offset tensor is already rank 1, so the tensor.collapse_shape branch
      is skipped. See ptr_gather_offset_collapse.py for the other side of this
      same `if`.

WHAT PROVES IT (01-structured.mlir)
      %2 = "tts.load"(%1) : (tensor<128x!tt.ptr<i32>>) -> tensor<128xi32>
      %3 = tts.make_gather_scatter_tptr %arg0 to sizes: [128]
           gather_scatter_dim: 0 gather_scatter_offset: %2, strides: [1],
           offsets: [0] : tensor<128xi32>  <f32> to !tt.ptr<tensor<128xf32>>
      %4 = "tts.load"(%3) <{static_mask_dims = array<i64>}>
           : (!tt.ptr<tensor<128xf32>>) -> tensor<128xf32>
  The gather offset operand IS the loaded tensor %2, verbatim -- no arithmetic
  at all between the index load and the pointer op, which is exactly what
  rebuild-unsupported-success means. There is no tensor.collapse_shape anywhere
  in the module, and no `gather_scatter_mask:` clause.

torch equivalent: o = x[i]

Base case for the ptr_gather_* family: x[i] -> tts.make_gather_scatter_tptr with the raw index tensor as offset.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 4096
BLOCK = 128
M = 64          # index range


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_1d_kernel(x_ptr, i_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(i_ptr + r)
        tl.store(o_ptr + r, tl.load(x_ptr + i))

    return ptr_gather_1d_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "i": torch.randint(0, M, (BLOCK,), generator=g, dtype=torch.int32)}


def reference(inputs):
    return {"o": inputs["x"][inputs["i"].long()]}


SPEC = KernelSpec(
    name="ptr_gather_1d_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("i", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
