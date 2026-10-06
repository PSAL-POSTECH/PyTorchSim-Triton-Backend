"""A mask MaskAnalysis CAN parse: it becomes the `dims` clamp of tts.load/store.

SIBLING FILE: tests/cases/mask_unparsed_to_gather.py, which is the SAME
kernel with the mask changed from `r < n` (affine, parseable) to `mv < 5` (mv
comes out of memory, unparseable). Read the two together -- that one comparison
is the whole reason MaskAnalysis exists, and the two files differ by one token
in the kernel and by everything in the output IR.

COVERS (4)
  mask-empty-passthrough          PtrAnalysis.cpp:47
      `masks.empty()` in applyUnstructuredMask -> `return ptr` unchanged.
      MaskState::getUnstructuredMasks() came back empty because parseCmp
      absorbed the whole mask into dims. This is the only exit from
      applyUnstructuredMask that leaves a plain tts.make_tptr alone.
  load-mask-applied-dims          PtrAnalysis.cpp:1804
  store-mask-applied-dims         PtrAnalysis.cpp:1947
      rewriteLoadOp / rewriteStoreOp: mask present, parse ok, applyUnstructuredMask
      ok -> `dims = mstate.dims`.
  load-other-scalar               PtrAnalysis.cpp:1810
      op.getOther() present and tts::utils::getScalarValue succeeded, so tts.load
      gets a third operand.

WHAT PROVES IT (01-structured.mlir, verbatim)
    %4  = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [0],
                        shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
    %7  = arith.minsi %c128, %6 : index          <- %6 = index_cast of n
    %8  = arith.maxsi %7, %c0 : index
    %cst_0 = arith.constant -1.000000e+00 : f32
    %9  = "tts.load"(%4, %8, %cst_0) <{operandSegmentSizes = array<i32: 1, 1, 1>,
          static_mask_dims = array<i64: -9223372036854775808>}>
    "tts.store"(%11, %9, %15) <{static_mask_dims = array<i64: -9223372036854775808>}>
  Three things at once: the ptr operand is the ORIGINAL tts.make_tptr, unchanged
  (-> mask-empty-passthrough); operand #2 is a `dims` entry, the
  index_cast/minsi/maxsi clamp max(min(128,n),0) (-> load-mask-applied-dims on
  the load, store-mask-applied-dims on the store); operand #3 of the load is
  %cst_0, the de-splatted scalar `other` (-> load-other-scalar). The store has
  no third operand -- tts.store has no `other` field at all. The
  -9223372036854775808 in static_mask_dims is ShapedType::kDynamic, i.e. "this
  dim's size is the SSA operand", not a size.
  Zero tts.make_gather_scatter_tptr, zero tts.gather in the whole module.

WHY IT MATTERS
  This is the baseline every other file in the mask_* family is a deviation
  from. `dims` is a size clamp: the emitted memref.copy simply copies fewer
  elements. Nothing per-element survives -- which is exactly why an unparseable
  mask cannot be expressed this way and has to become a gather instead.

torch equivalent: o[:n] = x[:n]  (the `other=-1.0` lanes are masked off on the
store too, so they are never written).

Parseable mask -> tts.load `dims`. Contrast with mask_unparsed_to_gather.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
NLIM = 96           # runtime `n`, deliberately not a multiple of BLOCK


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_structured_dims_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + r, mask=r < n, other=-1.0)
        tl.store(o_ptr + r, v, mask=r < n)

    return mask_structured_dims_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    o = torch.zeros(BLOCK, dtype=torch.float32)
    o[:NLIM] = inputs["x"][:NLIM]
    return {"o": o}


SPEC = KernelSpec(
    name="mask_structured_dims_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t")], "scalar_values": {"n": NLIM}},
)
