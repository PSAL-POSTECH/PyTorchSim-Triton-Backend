"""A gather offset that arrives 2-D: tensor.collapse_shape to rank 1.

tts.make_gather_scatter_tptr's verifier requires gather_scatter_offset to be a
RANK-1 integer tensor. A loaded index tensor does not have to be rank 1 --
here it is 8x1 -- so createTTSMakeGatherScatterTensorPtrOp flattens it first.

SIBLING: tests/cases/ptr_gather_1d.py is the `else` of this exact `if`
(PtrAnalysis.cpp:686): its offset is already rank 1 and no collapse appears.
The two files together are the whole branch.

COVERS (2)
  gs-offset-collapse-shape   PtrAnalysis.cpp:686
      `cast<ShapedType>(nonContinuousOffset.getType()).getRank() > 1` ->
      tensor::CollapseShapeOp to 1-D (size = product of dims); state.offsets[dim]
      is mutated to the collapsed value.
  tts-gs-offset-collapsed    TritonStructuredDialect.td:172
      the ODS side of the same fact: gather_scatter_offset is declared
      TT_IndexTensorLike and the verifier (TritonStructuredOps.cpp:216-234)
      rejects rank != 1, which is WHY the collapse exists.
      (.td entries are TableGen, so gdb cannot break on them --
      the branch census reports them as unmeasurable, not as missed.
      The evidence for this one is the emitted assembly quoted below.)

WHAT PROVES IT (01-structured.mlir, verbatim)
      %6 = "tts.load"(%5) : (tensor<8x1x!tt.ptr<i32>>) -> tensor<8x1xi32>
      %collapsed = tensor.collapse_shape %6 [[0, 1]]
                   : tensor<8x1xi32> into tensor<8xi32>
      %8 = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 1]
           gather_scatter_dim: 0 gather_scatter_offset: %collapsed,
           strides: [1, 0], offsets: [0, 0]
           : tensor<8xi32>  <f32> to !tt.ptr<tensor<8x1xf32>>
  The `tensor.collapse_shape ... [[0, 1]]` is the branch. Note the POINTER stays
  8x1 (`sizes: [8, 1]`) -- only the offset operand is flattened, so the result
  type of the load is still tensor<8x1xf32>. Nothing else in the module contains
  a collapse_shape, and grepping for one is the cheapest check that this file
  still does its job.

WHY THE INDEX IS 8x1 AND NOT 8x2
  The gather dim must be the ONLY dim with extent > 1: rebuildAsUnsupportedOp
  rejects anything else (isNotSingleDim). A 2-D index whose second dim is also
  > 1 is refused instead of collapsed -- that case is
  tests/cases/ptr_gather_multidim_refused.py, load 1. So "2-D offset" and
  "2-D data" are different things, and only the former reaches this branch.

torch equivalent: o[j] = x[i2[j, 0]] for j in range(8), where i2 is the 8x1
index block read from i[0::2].

Rank-2 gather offset -> tensor.collapse_shape before tts.make_gather_scatter_tptr.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
NI = 64
BM = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_offset_collapse_kernel(x_ptr, i_ptr, o_ptr,
                                          BM: tl.constexpr):
        rm = tl.arange(0, BM)
        r1 = tl.arange(0, 1)
        # an 8x1 index block: rank 2, but only dim 0 has extent > 1
        i2 = tl.load(i_ptr + rm[:, None] * 2 + r1[None, :])
        a = tl.load(x_ptr + i2)
        tl.store(o_ptr + rm[:, None] * 1 + r1[None, :], a)

    return ptr_gather_offset_collapse_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, NX, (NI,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    x, i = inputs["x"], inputs["i"].long()
    rows = i[0:2 * BM:2]                 # i2[:, 0]
    return {"o": x[rows]}


SPEC = KernelSpec(
    name="ptr_gather_offset_collapse_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr"},
    constexprs={"BM": BM},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (NI,)),
          Arg("o", OUT, "float32", (BM,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
