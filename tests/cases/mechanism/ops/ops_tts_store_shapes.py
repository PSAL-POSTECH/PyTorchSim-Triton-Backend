"""Every operand slot of tts.store -- and the two ways it is NOT tts.load.

Split out of the old tests/cases/ptr_structured.py section F. Three stores
to disjoint slices of the same output, differing only in the mask. The point of
keeping them apart from ops_tts_load_shapes.py is the asymmetry: TTS_StoreOp
has no `other` operand and no results, and (unlike LoadOp) it carries no
AttrSizedOperandSegments attribute at all, so its printed form is a different
shape and anything written against the load's shape will misread it.

NOT GDB-MEASURABLE: these are ODS facts in TritonStructuredDialect.td, not C++
branches, so the branch census classifies them as ".td, no code". The
evidence is the printed op:
    python tools/lower_steps.py tests/cases/ops_tts_store_shapes.py
    grep 'tts.store' \\
        work/ops_tts_store_shapes_kernel/triton-shared-detail/01-structured.mlir

COVERS (4)   include/triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.td

  tts-store-no-mask            :362  mask_dims and static_mask_dims both empty
                                     -> hasMask() is false. Store 1.
  tts-store-mask-dims-static   :349  static_mask_dims holds a real constant and
                                     mask_dims is empty. Store 2.
  tts-store-mask-dims-dynamic  :348  the mask_dims variadic is present and
                                     appended AFTER $value, with kDynamic in
                                     the attribute. Store 3.
  tts-store-no-other-field     :346  the structural asymmetry itself: the
                                     argument list is (ptr, value, mask_dims,
                                     static_mask_dims) with no `other` and no
                                     results. Every store here shows it -- none
                                     of the three has a third scalar operand,
                                     and all end in `-> ()`.

WHAT PROVES IT FIRED (01-structured.mlir)

  store 1, unmasked -- note there is NO operandSegmentSizes attribute, which by
  itself distinguishes a tts.store from a tts.load:
      "tts.store"(%6, %4) <{static_mask_dims = array<i64>}>
          : (tensor<32x!tt.ptr<f32>>, tensor<32xf32>) -> ()
  store 2, constant bound -> the clamp is a literal in the attribute:
      "tts.store"(%10, %4) <{static_mask_dims = array<i64: 16>}>
          : (tensor<32x!tt.ptr<f32>>, tensor<32xf32>) -> ()
  store 3, runtime bound -> a third operand that is an `index`, in the position
  where a load would carry `other`:
      "tts.store"(%15, %4, %19) <{static_mask_dims = array<i64:
          -9223372036854775808>}>
          : (tensor<32x!tt.ptr<f32>>, tensor<32xf32>, index) -> ()

torch equivalent: see reference(). Runtime scalar nfull=32.

ODS shapes of tts.store: no mask, static mask dim, dynamic mask dim, and the missing `other` slot. Verified by reading 01-structured.mlir. Runtime i32 scalar nfull=32.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 32
NO = 3 * BLOCK
KEEP = 16
NFULL = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_tts_store_shapes_kernel(x_ptr, o_ptr, nfull, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + r)

        tl.store(o_ptr + r, v)                                   # 1 no mask
        tl.store(o_ptr + (BLOCK + r), v, mask=r < 16)            # 2 static dim
        tl.store(o_ptr + (2 * BLOCK + r), v, mask=r < nfull)     # 3 dynamic dim

    return ops_tts_store_shapes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    v = inputs["x"][0:BLOCK]
    o = torch.zeros(NO, dtype=torch.float32)
    o[0:BLOCK] = v
    o[BLOCK:BLOCK + KEEP] = v[:KEEP]
    o[2 * BLOCK:2 * BLOCK + NFULL] = v[:NFULL]
    return {"o": o}


SPEC = KernelSpec(
    name="ops_tts_store_shapes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "nfull": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (NO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"nfull": NFULL}},
)
