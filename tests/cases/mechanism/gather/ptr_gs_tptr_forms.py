"""Every operand form tts.make_gather_scatter_tptr can be printed in.

This file is about the OP, not about an analysis: TritonStructuredDialect.td
declares gather_scatter_offset as `TT_IndexTensorLike` (i32 OR i64 tensor),
gather_scatter_dim as a plain I32Attr (0 or 1 or 2 -- a real dispatch
difference downstream), strides/offsets as DynamicIndexLists (each slot either
an SSA value or a literal), and the mask as Optional. Six loads, one table, one
line each: the point is to have every alternative of that syntax present in one
module so the printed assembly can be read as a legend.

Note this op has NO `shape:` and NO `order:` field. A gather pointer can never
be a wrap-around or block pointer; compare tts.make_tptr, which always prints
both.

COVERS (12)   lib/Dialect/TritonStructured/IR/TritonStructuredDialect.td
  These are TableGen op definitions, NOT C++ statements, so gdb cannot break on
  them: the branch census counted them as "not measurable (.td, no code)"
  rather than as covered or missed. The evidence is the emitted assembly quoted
  below, which is why every clause is quoted verbatim.
  tts-gs-tptr-created               .td:132   the op exists at all.
  tts-gs-offset-i32                 .td:173   `gather_scatter_offset: %N ...
                                              : tensor<8xi32>`   [load 1]
  tts-gs-offset-i64                 .td:173   the I64Tensor alternative of
                                              TT_IndexTensorLike  [load 2]
  tts-gs-strides-offsets-dyn-static .td:176   `strides: [%c32, 1]` (literal)
                                              vs `strides: [%N, 1]` (SSA)
                                              [loads 1 and 3]
  tts-gs-dim-selection              .td:174   `gather_scatter_dim: 0` vs
                                              `gather_scatter_dim: 1`
                                              [loads 1 and 4]
  tts-gs-mask-absent                .td:180   no `gather_scatter_mask:` clause
                                              and one fewer trailing type
                                              [loads 1-4]
  tts-gs-mask-present               .td:180   the optional clause present
                                              [load 5, and the masked store]
  tts-load-ptr-is-blockptr          .td:313   tts.load's $ptr is
                                              !tt.ptr<tensor<8x16xf32>>, the
                                              block-pointer form, because it
                                              came from this op
  tts-load-mask-dim-zero-gather     .td:315   a LITERAL 0 in the tts.load's
                                              static_mask_dims -- "this dim's
                                              mask is on the pointer op"; must
                                              not be read as "load nothing"
  tts-store-mask-dim-zero-gather    .td:349   the same on a tts.store
  tts-store-ptr-blockptr-vs-tensor  .td:346   both alternatives are present in
                                              this module: the masked store's
                                              ptr is !tt.ptr<tensor<8x16xf32>>,
                                              the plain store's is
                                              tensor<8x16x!tt.ptr<f32>>
  tts-gs-offset-collapsed           .td:172 is the one alternative NOT here --
      it has its own file, tests/cases/ptr_gather_offset_collapse.py.

WHAT PROVES IT (01-structured.mlir, one line per load, verbatim)
  1  %29 = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 16]
       gather_scatter_dim: 0 gather_scatter_offset: %12,
       strides: [%c32_5, 1], offsets: [0, 0]
       : tensor<8xi32>  <f32> to !tt.ptr<tensor<8x16xf32>>
  2  %37 = ... gather_scatter_offset: %16, strides: [%c32_8, 1],
       offsets: [0, 0] : tensor<8xi64>  <f32> to !tt.ptr<tensor<8x16xf32>>
        ^ the ONLY difference from load 1 is i32 -> i64 on the offset operand.
  3  %45 = ... gather_scatter_offset: %12, strides: [%4, 1], ...
        ^ %4 is an index_cast of the loaded stride, so slot 0 is SSA where load
          1's was the constant %c32_5. Slot 1 stays the literal 1 -- the mixed
          static/dynamic list.
  4  %55 = ... gather_scatter_dim: 1 gather_scatter_offset: %20,
       strides: [%c32_9, 1], offsets: [0, 0] : tensor<16xi32> ...
        ^ the offset tensor is 16 long now, because the gathered axis is the
          COLUMNS.
  5  %65 = ... gather_scatter_dim: 0 gather_scatter_offset: %12
       gather_scatter_mask: %59, strides: [%c32_5, 1], offsets: [0, 0]
       : tensor<8xi32> tensor<8xi1> <f32> to !tt.ptr<tensor<8x16xf32>>
        ^ mask clause present AND an extra tensor<8xi1> in the type list
     %66 = "tts.load"(%65, %cst_11)
           <{operandSegmentSizes = array<i32: 1, 0, 1>,
             static_mask_dims = array<i64: 0, 12>}>
           : (!tt.ptr<tensor<8x16xf32>>, f32) -> tensor<8x16xf32>
        ^ the literal 0 in slot 0 -- and note slot 1's 12 IS a real size, from
          `rn < 12`. Same array, two meanings, told apart only by whether the
          pointer op carries a gather_scatter_mask.
  Every load also leaves a dead `sizes: [8, 1]` intermediate next to the live
  op (`strides: [%c32_4, 0]`): the state before the column term was added.
  The two stores print the two $ptr alternatives:
      "tts.store"(%72, %78) <{static_mask_dims = array<i64>}>
          : (tensor<8x16x!tt.ptr<f32>>, tensor<8x16xf32>) -> ()
      "tts.store"(%85, %31) <{static_mask_dims = array<i64: 0, 12>}>
          : (!tt.ptr<tensor<8x16xf32>>, tensor<8x16xf32>) -> ()

WHY IT MATTERS
  Reading a gather kernel's IR means reading this syntax, and several clauses
  are optional or positional -- a missing `gather_scatter_mask:` and a
  `static_mask_dims` slot of 0 mean OPPOSITE things about where the mask went.
  One module with all of them is the legend for every other file here.

torch equivalent: compile-time probe (`reference` returns zeros) -- the loads
exist to produce syntax, and load 3's stride comes from a buffer.

One module containing every alternative of the tts.make_gather_scatter_tptr / tts.load / tts.store syntax.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NT = 1024
NI = 64
TD = 32
BM = 8
BN = 16
KCOL = 12


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gs_tptr_forms_kernel(tbl_ptr, i32_ptr, i64_ptr, c_ptr, s_ptr,
                                 o_ptr, o2_ptr, TD: tl.constexpr,
                                 BM: tl.constexpr, BN: tl.constexpr,
                                 KCOL: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        stride = tl.load(s_ptr)               # a RUNTIME row stride
        nkeep = tl.load(s_ptr + 1)
        i32 = tl.load(i32_ptr + rm)           # tensor<8xi32>
        i64 = tl.load(i64_ptr + rm)           # tensor<8xi64>
        cidx = tl.load(c_ptr + rn)            # 16 column indices

        # 1. i32 offset, literal stride, dim 0, no mask
        v0 = tl.load(tbl_ptr + i32[:, None] * TD + rn[None, :])
        # 2. the same with an i64 offset tensor
        v1 = tl.load(tbl_ptr + i64[:, None] * TD + rn[None, :])
        # 3. the same with an SSA stride
        v2 = tl.load(tbl_ptr + i32[:, None] * stride + rn[None, :])
        # 4. the gathered axis is the columns: gather_scatter_dim 1
        v3 = tl.load(tbl_ptr + rm[:, None] * TD + cidx[None, :])
        # 5. with a mask MaskAnalysis cannot fold into a size
        mgs = (i32 < nkeep)[:, None] & (rn[None, :] < KCOL)
        v4 = tl.load(tbl_ptr + i32[:, None] * TD + rn[None, :], mask=mgs,
                     other=0.0)

        # the two tts.store $ptr forms, side by side
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :],
                 v0 + v1 + v2 + v3 + v4)
        tl.store(o2_ptr + i32[:, None] * TD + rn[None, :], v0, mask=mgs)

    return ptr_gs_tptr_forms_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.zeros(NI, dtype=torch.int32)
    idx[:BM] = torch.randperm(NT // TD, generator=g)[:BM].to(torch.int32)
    return {"tbl": torch.randn(NT, generator=g, dtype=torch.float32),
            "i32": idx,
            "i64": idx.to(torch.int64),
            "c": torch.randint(0, BN, (NI,), generator=g, dtype=torch.int32),
            "s": torch.tensor([TD, BM // 2], dtype=torch.int32)}


def reference(inputs):
    """What the five arms actually compute.

    This used to be `zeros(...)` -- a placeholder, because the kernel could not
    be lowered and the numbers were never reached. Written out arm by arm, in
    the same order the kernel adds them; `stride` is `s[0]` which make_inputs
    sets to TD, so arm 3 agrees with arm 1 by construction rather than by
    accident, and both are kept because the POINT is that a literal and an SSA
    stride take different paths to the same place.

    o2 is a scatter: `i32[:BM]` is a permutation slice, so no two rows collide
    and the store order does not matter.
    """
    import torch
    tbl, i32, i64 = inputs["tbl"], inputs["i32"], inputs["i64"]
    cidx, sv = inputs["c"], inputs["s"]
    stride, nkeep = int(sv[0]), int(sv[1])
    rm = torch.arange(BM)
    rn = torch.arange(BN)
    a = i32[:BM].long()[:, None]
    b = i64[:BM].long()[:, None]
    c = cidx[:BN].long()[None, :]

    v0 = tbl[a * TD + rn[None, :]]
    v1 = tbl[b * TD + rn[None, :]]
    v2 = tbl[a * stride + rn[None, :]]
    v3 = tbl[rm[:, None] * TD + c]
    mgs = (i32[:BM] < nkeep)[:, None] & (rn[None, :] < KCOL)
    v4 = torch.where(mgs, tbl[a * TD + rn[None, :]], torch.zeros(()))

    o = (v0 + v1 + v2 + v3 + v4).reshape(-1)
    o2 = torch.zeros(NT, dtype=torch.float32)
    flat = (a * TD + rn[None, :])
    o2[flat[mgs]] = v0[mgs]
    return {"o": o, "o2": o2}


SPEC = KernelSpec(
    name="ptr_gs_tptr_forms_kernel", kernel=kernel,
    signature={"tbl_ptr": "*fp32", "i32_ptr": "*i32", "i64_ptr": "*i64",
               "c_ptr": "*i32", "s_ptr": "*i32", "o_ptr": "*fp32",
               "o2_ptr": "*fp32", "TD": "constexpr", "BM": "constexpr",
               "BN": "constexpr", "KCOL": "constexpr"},
    constexprs={"TD": TD, "BM": BM, "BN": BN, "KCOL": KCOL},
    args=[Arg("tbl", IN, "float32", (NT,)), Arg("i32", IN, "int32", (NI,)),
          Arg("i64", IN, "int64", (NI,)), Arg("c", IN, "int32", (NI,)),
          Arg("s", IN, "int32", (2,)),
          Arg("o", OUT, "float32", (BM * BN,)),
          Arg("o2", OUT, "float32", (NT,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
