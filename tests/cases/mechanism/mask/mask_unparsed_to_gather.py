"""A mask MaskAnalysis CANNOT parse: the load is refused and falls out to a gather.

SIBLING FILE: tests/cases/mask_structured_dims.py. That kernel is this one
with `mask=mv < 5` replaced by `mask=r < n`. One token; two completely different
lowerings. Here `mv` is LOADED FROM MEMORY, so parseCmp cannot parse its lhs,
MaskState::parse fails, and rewriteLoadOp/rewriteStoreOp bail before they ever
build a tts.load. The tt.load survives pass 1 and --triton-to-unstructured then
turns it into a per-element tts.gather.

COVERS (2)
  load-mask-parse-fail            PtrAnalysis.cpp:1796
      rewriteLoadOp: `if (failed(mstate.parse(mask, loc, builder)))` -> remark
      "MaskAnalysis failed" and return failure. Nothing of the load is rewritten,
      not even the (perfectly structured!) pointer.
  store-mask-parse-fail           PtrAnalysis.cpp:1939
      the identical check in rewriteStoreOp. Same mask value, so the two
      failures are one construct seen from both sides.

WHAT PROVES IT
  With --debug-only=triton-ptr-analysis, twice:
      remark: MaskAnalysis failed
  In 01-structured.mlir the tt.addptr chain and both tt ops are still there,
  even though a tts.make_tptr for that very address had already been built:
      %7  = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [0],
                          shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
      %8  = tt.addptr %6, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %9  = tt.load %8, %5, %cst : tensor<128x!tt.ptr<f32>>
      tt.store %12, %9, %5 : tensor<128x!tt.ptr<f32>>
  (%7 and %11 are dead on arrival. The only tts.load that survives belongs to
  the mask load `m_ptr + r`, which has no mask of its own.)
  By 02-unstructured.mlir the refused pair has become
      %4 = tts.gather %arg0[%0] mask = %3 default = %cst
           : (<f32>, tensor<128xi32>) -> tensor<128xf32>
      tts.scatter %4 into %arg2[%0] mask = %3
           : tensor<128xf32> into (<f32>, tensor<128xi32>)
  -- the mask is now carried per element, and the contiguous copy is gone:
  05-unstruct-memref.mlir lowers the gather to a linalg.generic whose body is
  `scf.if %mask { tensor.extract } else { %cst }`, 128 independent element reads
  where the sibling file emitted one memref.copy.

WHY IT MATTERS
  The cost of an unparseable mask is not "a slower mask", it is losing the
  structured pointer entirely -- PtrAnalysis refuses the whole load, including
  the address, and everything downstream has to reconstruct it element by
  element. That is the single most important fact in this directory.

torch equivalent: o = where(m < 5, x[:128], 0.0) for the stored lanes; lanes
with m >= 5 are never written (the store is masked too).

Unparseable mask -> PtrAnalysis refuses the load -> tts.gather. Contrast with mask_structured_dims.py.
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
    def mask_unparsed_to_gather_kernel(x_ptr, m_ptr, o_ptr,
                                       BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        mv = tl.load(m_ptr + r)             # data dependent -> unparseable mask
        v = tl.load(x_ptr + r, mask=mv < 5, other=0.0)
        tl.store(o_ptr + r, v, mask=mv < 5)

    return mask_unparsed_to_gather_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "m": torch.randint(0, 10, (N,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    keep = inputs["m"][:BLOCK] < 5
    return {"o": torch.where(keep, inputs["x"][:BLOCK],
                             torch.zeros((), dtype=torch.float32))}


SPEC = KernelSpec(
    name="mask_unparsed_to_gather_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "m_ptr": "*i32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("m", IN, "int32", (N,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
