"""The rank-1 masks MaskAnalysis turns into a plain size clamp -- five
comparisons, five `static_mask_dims` shapes, no gather anywhere.

Split out of the old tests/cases/ptr_structured.py section C. A mask that
MaskState::parse understands never becomes a real mask in the output: it
becomes the SIZE of the load. Everything here ends as `tts.load %ptr[dim]`,
either with a constant in static_mask_dims or with one index operand computed
by a minsi/maxsi/subi chain. Five loads, one per parse dispatch arm that a 1-D
comparison can reach.

COVERS (18)   source root /workspace/triton_shared/, worktree lines
              (MaskAnalysis.cpp is lib/Analysis/MaskAnalysis.cpp)

  the parse() TypeSwitch, checked in this exact order (MaskAnalysis.cpp:33-57):
    dispatch-constant     :34   arith.constant is tried FIRST, before the
                                integer-type check -- loads 2 and 3.
    dispatch-addi         :38   a TENSOR arith.addi (a scalar one would have
                                been swallowed by the IntegerType arm) --
                                loads 1 and 3.
    dispatch-cmpi         :42   a TENSOR of i1 -- every load here.
    dispatch-makerange    :44   tt.make_range -- every load here.
    dispatch-splat        :48   tt.splat of the runtime bound -- loads 1/3/5.
    dispatch-extsi        :54   a TENSOR arith.extsi -- load 5.
  the leaves those arms produce:
    makerange-success     :713  stride == 1 -> start/end are IndexAttrs and
                                dims = [shape[0]], the constants that let the
                                later min/max fold.
    const-dense-splat     :291  a DenseElementsAttr splat constant is
                                materialized as an INDEX arith.constant Value
                                (not an Attribute), so getIntAttr on it returns
                                nullopt downstream.
    intscalar-indexcast   :315  an integer scalar of width != 1 ->
                                `arith.index_cast %n : i32 to index`.
    splat-integer-success :766  a non-i1 integer splat sets BOTH scalar and
                                full dims, and leaves masks EMPTY.
  addStates (:200):
    add-lhs-scalar        :217  only the LHS has a scalar: `pid*BLOCK + r`
                                lowers to addi(splat, range), so the splat is
                                on the left -- load 1.
    add-rhs-scalar        :219  only the RHS has one: `r + 4` lowers to
                                addi(range, const), the mirror image -- load 3.
                                Same result shape, different operand order into
                                addOFRs, which is the only reason both ids
                                exist.
  parseCmp (:459):
    cmp-shaped-single-nonunit-dim :464  exactly one dim has extent != 1 ->
                                cmpOpDim = that dim, and an unstructured-mask
                                candidate is registered before any validation.
    cmp-1d-mask-record    :485  rank == 1, so masks[0] is the cmp itself, with
                                no tensor.collapse_shape in front of it.
    cmp-newdim-slt-ult    :566  the slt/ult arm: minsi(end, bound) ->
                                maxsi(.., start) -> subi(.., start). The maxsi
                                is what stops a fully-masked-off range from
                                going negative -- loads 1, 2, 3, 5.
    cmp-newdim-sge-zero   :584  the sge-against-0 arm: newDim is lhs.dims[cmpDim]
                                verbatim, i.e. the comparison is ASSUMED TRUE
                                and produces a full static dim -- load 4.
    cmp-success-clears-mask :596 on success masks[cmpOpDim] is set back to
                                nullptr, so getUnstructuredMasks() is empty and
                                PtrAnalysis keeps the plain tts.make_tptr.
  and the consequence in PtrAnalysis:
    unstructured-none     PtrAnalysis.cpp:47   applyUnstructuredMask sees an
                                empty mask list and returns the pointer
                                unchanged.

WHAT PROVES IT FIRED (01-structured.mlir)

  load 1, runtime bound -> ONE index operand, kDynamic in the attr, and the
  whole minsi/maxsi/subi chain cmp-newdim-slt-ult describes:
      %12 = arith.addi %c32, %11 : index          (end = start + 32)
      %13 = arith.index_cast %arg2 : i32 to index (the bound n)
      %14 = arith.minsi %12, %13 : index
      %15 = arith.maxsi %14, %11 : index
      %16 = arith.subi %15, %11 : index
      %17 = "tts.load"(%7, %16, %cst_3) <{operandSegmentSizes = array<i32: 1,1,1>,
            static_mask_dims = array<i64: -9223372036854775808>}>
  load 2, constant bound -> no index operand at all, minOFRs folded it:
      %19 = "tts.load"(%7, %cst_4) <{operandSegmentSizes = array<i32: 1, 0, 1>,
            static_mask_dims = array<i64: 24>}>
  load 4, `>= 0` -> the FULL extent, and no arith.minsi/maxsi is emitted for it
  at all:
      static_mask_dims = array<i64: 32>
  load 5 produces the same operand shape as load 1 -- the arith.extsi is left
  dead in the TTIR and never reaches MaskState.
  Nothing in the module is a tts.make_gather_scatter_tptr: that absence is what
  cmp-success-clears-mask and unstructured-none mean.

torch equivalent: see reference(); n is a runtime scalar, 20 here.

Every rank-1 mask MaskAnalysis accepts, one load each. Runtime i32 scalar n=20. Numerically runnable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 32
NVAL = 20       # the runtime bound `n`
CONSTBOUND = 24


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_1d_cmp_dims_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        offs = pid * BLOCK + r
        mp = x_ptr + offs

        v = tl.load(mp, mask=offs < n, other=0.0)          # 1 runtime bound
        v += tl.load(mp, mask=r < 24, other=0.0)           # 2 constant bound
        offc = r + 4                                       # 3 range + constant
        v += tl.load(mp, mask=offc < n, other=0.0)
        v += tl.load(mp, mask=offs >= 0, other=0.0)        # 4 assumed always true
        o64 = r.to(tl.int64)                               # 5 i64 offsets
        v += tl.load(mp, mask=o64 < n.to(tl.int64), other=0.0)
        tl.store(o_ptr + r, v)

    return mask_1d_cmp_dims_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    r = torch.arange(BLOCK)
    base = x[0:BLOCK]                       # pid == 0
    z = torch.zeros(BLOCK, dtype=torch.float32)
    v = torch.where(r < NVAL, base, z)
    v = v + torch.where(r < CONSTBOUND, base, z)   # mask is on r, not offs
    v = v + torch.where(r + 4 < NVAL, base, z)
    v = v + base                            # `>= 0` is always true
    v = v + torch.where(r < NVAL, base, z)
    return {"o": v}


SPEC = KernelSpec(
    name="mask_1d_cmp_dims_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"n": NVAL}},
)
