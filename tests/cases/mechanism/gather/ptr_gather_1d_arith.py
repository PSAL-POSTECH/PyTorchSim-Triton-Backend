"""Arithmetic ON a 1-D gather offset: addState/mulState's unstructured arms.

SIBLING: tests/cases/ptr_gather_1d.py is the same shape with NO arithmetic
(`x[i]`). Read that first -- it explains how a loaded index becomes a
non-structured PtrState at all. This file is what happens when you then add to
or multiply that state, which is where PtrAnalysis's two-pass design becomes
visible: the FIRST (structured) pass refuses to emit address arithmetic for a
non-structured dim and just carries the raw tensor, and a SECOND pass with
isAnalysisingUnstructured=true redoes the same expression and actually folds
strides in.

Four loads, one address shape (rank 1, gather dim 0), differing only in what is
combined with what -- that is the point: the four arms are one `if/else` chain
in addState and reading them side by side is the only way to see which operand
shape picks which.

COVERS (7)   /workspace/triton_shared/lib/AnalysisStructured/
  mulstate-unstruct-propagate         PtrAnalysis.cpp:596
      mulState, dim not structured, FIRST pass -> offset AND stride propagated
      unchanged (stride stays 1).  [load 1]
  mulstate-unstruct-mul-stride        PtrAnalysis.cpp:585
      the same dim in the SECOND pass -> offset kept verbatim, stride *= 4,
      shape = 0.  [load 1]
  addstate-unstruct-propagate-offset-structpass   PtrAnalysis.cpp:405
      addState, one side non-structured, FIRST pass -> stride forced to index 1
      and offset = the unstructured side. No arith is emitted in pass 1.
      [loads 1-4]
  addstate-unstruct-mul-offsets       PtrAnalysis.cpp:355
      second pass, offsets differ AND strides differ -> both offsets are scaled
      by their own stride and both strides forced to 1.  [load 2]
      NOTE: scaling by the stride is only correct for the UNSTRUCTURED side,
      whose offset is already a per-element tensor. A structured side keeps its
      index implicit (offset + stride*k), so `offset * stride` folded the whole
      range to 0 and load 2 silently became load 1. triton_shared c19a6f1
      (`PtrAnalysis: keep a structured term when merging it with an
      unstructured one`) materialises that index; 커밋 히스토리
      has the long version.
  addstate-unstruct-equal-offsets-add-strides     PtrAnalysis.cpp:387
      offsets equal (the same %i tensor on both sides), strides differ -> no
      arith on the offset tensor at all, the two strides are added.  [load 3]
  addstate-unstruct-stride-zero-normalize         PtrAnalysis.cpp:335
      a side whose stride is the constant 0 (the splatted literal 7) gets that
      stride rewritten to index 1 before the merge.  [load 4]
  addstate-unstruct-equal-strides-add-offsets     PtrAnalysis.cpp:370
      ... and now both strides are 1, so the arm below fires: offsets added,
      the scalar one expanded to tensor shape first.  [load 4]

WHAT PROVES IT (01-structured.mlir)
  Load 1  x[i*4]     -- stride folded into the OP, not into the tensor:
      %2 = "tts.load"(%1) : (tensor<128x!tt.ptr<i32>>) -> tensor<128xi32>
      %3 = tts.make_gather_scatter_tptr %arg0 to sizes: [128]
           gather_scatter_dim: 0 gather_scatter_offset: %2, strides: [4],
           offsets: [0] : tensor<128xi32> <f32> to !tt.ptr<tensor<128xf32>>
    `strides: [4]` with the RAW loaded %2 as the offset. That 4 exists only
    because of the second pass (:585); the first pass had left stride 1 (:596).
  Load 2  x[i*4 + r*8] -- offsets differ and strides differ, so :355 scales
    BOTH sides into the offset tensor and leaves stride 1:
      %12 = arith.index_cast %c4_4 : index to i32
      %13 = tt.splat %12 : i32 -> tensor<128xi32>
      %14 = arith.muli %4, %13 : tensor<128xi32>      (the loaded index * 4)
      %15 = arith.index_cast %c0 : index to i32
      %16 = tt.splat %15 : i32 -> tensor<128xi32>     (the arange side * ... 0)
      %17 = arith.addi %14, %16 : tensor<128xi32>
      %18 = tts.make_gather_scatter_tptr %arg0 ... gather_scatter_offset: %17,
            strides: [1], offsets: [0]
    Read %16 carefully: the arange side's PtrState is (offset 0, stride 8), and
    :355 multiplies each side's OFFSET by its own stride -- 0 * 8 -- then forces
    both strides to 1. The `+ r*8` term is therefore GONE from the address.
    04-structured-memref.mlir confirms it, `offset: [%extracted], strides: [1]`
    where %extracted comes from %17. A gather PtrState has room for exactly one
    offset tensor per dim, and this arm spends it on the unstructured side.
    That is why `reference` below is a compile-time probe: this kernel's address
    is not the address the Python says.
  Load 3  x[i*4 + i*8] -- SAME offset tensor on both sides, so :387 takes the
    other branch: no tensor arith is emitted, the strides are added at index
    level and constant-folded:
      ... gather_scatter_offset: %2, strides: [12], offsets: [0]
    %2 is again the bare loaded index. 12 = 4 + 8.
  Load 4  x[i + 7]  -- the literal is a splat whose PtrState has stride 0;
    :335 promotes it to 1, then :370 adds the offsets:
      %c7 = arith.constant dense<7> : tensor<128xi32>
      %11 = arith.addi %2, %c7 : tensor<128xi32>
      ... gather_scatter_offset: %11, strides: [1], offsets: [0]

WHY IT MATTERS
  Loads 2 and 3 have almost the same source text and produce structurally
  different IR -- one materialises an arith.muli + arith.addi over 128 elements
  per address, the other materialises nothing and moves a constant into the
  pointer op. The discriminator is not "is it a multiply", it is whether the two
  sides' PtrStates happen to share an offset Value. That is the single least
  obvious thing about addState, and it is only legible with the four side by
  side.

torch equivalent: none -- compile-time probe (`reference` returns zeros). Load 2
does not compute the address its source line says (see above), so a torch
expression here would be a lie. Loads 1/3/4 alone would be
x[i*4] + x[i*12] + x[i+7].

addState/mulState unstructured arms: four ways to combine a gather offset with something else, one file so the arms are comparable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
NI = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_1d_arith_kernel(x_ptr, i_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(i_ptr + r)                  # the non-structured offset

        # 1. mulState only: stride 4 ends up ON THE POINTER OP
        a = tl.load(x_ptr + i * 4)
        # 2. offsets differ AND strides differ -> both scaled into the tensor
        b = tl.load(x_ptr + (i * 4 + r * 8))
        # 3. SAME offset tensor both sides -> strides added, no tensor arith
        c = tl.load(x_ptr + (i * 4 + i * 8))
        # 4. rhs is a splatted literal (stride 0 -> normalized to 1) -> offsets
        #    added
        d = tl.load(x_ptr + (i + 7))

        tl.store(o_ptr + r, a + b + c + d)

    return ptr_gather_1d_arith_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # every address below is <= i*12 + r*8 + 7, so cap i so nothing leaves x.
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, 128, (NI,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    # A real reference now. This used to be zeros because load 2's lowered
    # address dropped the `+ r*8` term, so the emitted kernel was not the Python
    # program and there was nothing honest to predict. With the structured term
    # preserved, all four loads compute what the source says.
    x, i = inputs["x"], inputs["i"].long()
    r = torch.arange(BLOCK)
    return {"o": x[i * 4] + x[i * 4 + r * 8] + x[i * 12] + x[i + 7]}


SPEC = KernelSpec(
    name="ptr_gather_1d_arith_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (NI,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
