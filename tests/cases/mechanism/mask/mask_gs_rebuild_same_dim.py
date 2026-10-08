"""The pointer is ALREADY a gather pointer and the unstructured mask lands on
the same dim -- so the op is rebuilt, not converted.

THE AXIS DECIDES, and these three files are the three answers it gives:
  mask_gs_from_structured.py  ptr is a tts.make_tptr, so there is no gathered
                              axis at all -> the mask rides on the ACCESS
  THIS FILE                   ptr is a tts.make_gather_scatter_tptr and the mask
                              is ON its gather_scatter_dim, so the mask IS the
                              partial gather -> re-create the op with the same
                              operands plus the mask
  mask_gs_dim_mismatch.py     same pointer, mask on another dim, which the
                              pointer's rank-1 slot cannot hold -> the ACCESS

  This file used to say the first line was "synthesize a gather offset (splat +
  make_range) and build a NEW op". That was the DEMOTION, and it is deleted.

COVERS (1)
  mask-gs-rebuild-with-mask       rebuildGatherWithMask
      The ptr's defining op is a tts::MakeGatherScatterTensorPtrOp whose
      gatherScatterDim equals the dim of the single unstructured mask, so a
      second MakeGatherScatterTensorPtrOp is created from its base / offset /
      dim / sizes / strides / offsets with `unstructuredMask` inserted as the
      third operand. Both rewriteLoadOp and rewriteStoreOp reach it now; a null
      result means "the pointer cannot say this", not a failure.

WHAT PROVES IT (01-structured.mlir, verbatim)
  Two tts.make_gather_scatter_tptr of the same shape, differing ONLY in the mask:
      %20 = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 16]
            gather_scatter_dim: 0 gather_scatter_offset: %5,
            strides: [%c16_3, 1], offsets: [0, 0]
            : tensor<8xi32>  <f32> to !tt.ptr<tensor<8x16xf32>>
      %23 = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 16]
            gather_scatter_dim: 0 gather_scatter_offset: %5
            gather_scatter_mask: %10, strides: [%c16_3, 1], offsets: [0, 0]
            : tensor<8xi32> tensor<8xi1> <f32> to !tt.ptr<tensor<8x16xf32>>
      %24 = "tts.load"(%23, %cst_4) <{operandSegmentSizes = array<i32: 1, 0, 1>,
            static_mask_dims = array<i64: 0, 16>}>
  %20 is what createTTSMakeGatherScatterTensorPtrOp built while analysing the
  address; %23 is applyUnstructuredMask's rebuild, with %10 = `arith.cmpi slt,
  %9, %cst_1` (the loaded row flags) spliced in. Only %23 has a consumer; %20
  and the rank-2 8x1 intermediate %15 are dead and survive until
  --remove-dead-values in the cleanup step. tts.load's static_mask_dims has the
  literal 0 in slot 0, the gather dim.

WHY IT MATTERS
  This is the only place in the linalg lowering where an already-emitted tts op is
  rebuilt rather than mutated, and the reason the dead first copy exists at all.
  If you are counting tts.make_gather_scatter_tptr in a module to attribute
  gathers to source lines, a masked gather contributes TWO.

torch equivalent: o[j, :] = where(m[j] < 5, x[i[j]*16 : i[j]*16+16], 0.0).

Gather ptr + unstructured mask on the gather dim -> the op is rebuilt with a mask operand (two make_gather_scatter_tptr, one dead).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BM = 8
BN = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_gs_rebuild_same_dim_kernel(x_ptr, i_ptr, m_ptr, o_ptr,
                                        BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        i = tl.load(i_ptr + rm)                 # gather dim is 0
        um = tl.load(m_ptr + rm) < 5            # unstructured mask, also dim 0
        v = tl.load(x_ptr + i[:, None] * BN + rn[None, :],
                    mask=um[:, None], other=0.0)
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], v)

    return mask_gs_rebuild_same_dim_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "i": torch.randint(0, N // BN, (BM,), generator=g, dtype=torch.int32),
            "m": torch.randint(0, 10, (BM,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    x, i, m = inputs["x"], inputs["i"].to(torch.int64), inputs["m"]
    rows = x[(i[:, None] * BN + torch.arange(BN)[None, :])]
    return {"o": torch.where((m < 5)[:, None], rows,
                             torch.zeros((), dtype=torch.float32)).reshape(-1)}


SPEC = KernelSpec(
    name="mask_gs_rebuild_same_dim_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "m_ptr": "*i32",
               "o_ptr": "*fp32", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (N,)), Arg("i", IN, "int32", (BM,)),
          Arg("m", IN, "int32", (BM,)), Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
