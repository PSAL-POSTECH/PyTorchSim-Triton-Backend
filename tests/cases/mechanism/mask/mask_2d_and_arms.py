"""How MaskAnalysis MERGES two masks: parseAnd's four arms.

Every 2-D masked load in Triton is an `&` of two masks, and `MaskState::parseAnd`
folds them into one state. A MaskState holds a mask TWICE OVER, and which half
an operand lands in is the whole question: `dims[i]` is a per-axis SIZE clamp (a
box, free -- the descriptor clamps) and `masks[i]` is a per-axis i1 TENSOR
(unstructured, paid for -- a select on a read, a park on a write).

SIBLING: tests/cases/mask_2d_cmp_expand.py parses a SINGLE mask. Read it first
if `dims` and `masks` are not yet distinct in your head.

ONE AXIS IS DESCRIBED BY ONE OF THE TWO, NEVER BOTH, and that invariant is what
the arms below are really about. MaskAnalysis states it outright when an axis
takes an i1:

    // Recover dims for unstructured mask dim to allow other dim to be processed.
    dims[dim] = builder.getIndexAttr(srcType.getShape()[dim]);

COVERS (4)   lib/AnalysisStructured/MaskAnalysis.cpp
  and-unstructured-vs-full-structured    MaskAnalysis.cpp:404   [load 1]
      an i1 on dim 0 against a dims[0] that is a CONSTANT equal to the full
      extent. `structuredMaskToUnstructuredMask` returns null for a full mask,
      so the i1 goes through untouched and nothing is synthesized.
  and-unstructured-vs-partial-structured MaskAnalysis.cpp:407   [load 2]
      the same axis, but the other side's dims[0] is `min(mbound, 8)` -- a
      runtime Value. A comparison is SYNTHESIZED (make_range, index_castui,
      splat, cmpi ult) and and-ed into the i1, and dims[0] goes back to the full
      8. The clamp is not kept beside the i1; that is the invariant, not an
      oversight, and the synthesis is its price.
  and-both-unstructured-same-dim         MaskAnalysis.cpp:437   [load 3]
      two i1s on dim 0 -> one arith.andi. Still ONE unstructured mask.
  and-multi-unstructured-fail            MaskAnalysis.cpp:441   [load 4]
      one i1 on dim 0 and another on dim 1 -> `getUnstructuredMasks().size() > 1`
      and parseAnd returns failure.

WHAT PROVES IT (03-linalg.mlir). All five 2-D accesses go through a PLAIN
`tts.make_tptr`; there is no `make_gather_scatter_tptr` in this kernel at all.

  Loads 1, 2 and 3 are identical in shape --
      "tts.load"(%ptr, %extent, %other, %m)
          <{static_mask_dims = array<i64: 8, -9223372036854775808>}>
          : (tensor<8x16x!tt.ptr<f32>>, index, f32, tensor<8x16xi1>)
    dim 0 is the full 8 (its truth is in the i1) and dim 1 is the ordinary
    min(n, 16) clamp. What differs is only how much IR built `%m`: load 1 hands
    over the loaded flags unchanged, load 3 is one `andi` of two flag vectors,
    and load 2 pays for the synthesized comparison.
  THE STORE IS THE CASE WHERE BOTH HALVES SURVIVE, because its box and its i1
  are on DIFFERENT axes -- `(om < mbound) & mn[None, :]` --
      "tts.store"(%ptr, %val, %66, %79)
          <{static_mask_dims = array<i64: -9223372036854775808, 16>}>
    and `%66` IS `min(mbound, 8)`, the same clamp load 2 had to give up. The
    axis decides: on its own axis the box is free, sharing an axis with an i1 it
    is demoted.
  Load 4 leaves the structured world: one `tts.gather`, a per-element index and
  the 2-D `andi` of the two broadcast masks --
      "tts.gather"(%base, %offsets, %mask, %other)
          : (!tt.ptr<f32>, tensor<8x16xi32>, tensor<8x16xi1>, f32)

WHY IT MATTERS. Loads 1 and 2 differ only in whether the co-occurring bound is a
compile-time constant, and that difference costs four extra ops and a clamp.
Load 4 is not "slower": it leaves the structured world, so the DMA moves one
element per descriptor.

AND THE REASON FOR ARM 4 HAS GONE STALE, which is worth knowing before anyone
argues from it. This file used to say the refusal is because "only one gather
dim exists" -- true when an i1 rode on `make_gather_scatter_tptr`'s mask slot,
one bit per ROW. It does not any more: the mask arrives as a full-tile
`tensor<8x16xi1>` operand on a PLAIN tptr, which has no axis. Two per-axis i1s
could be broadcast and and-ed into one, and that is machinery this very kernel
already uses -- load 2 broadcasts a `tensor<8xi1>` to 8x16 before its `andi`.
The guard's own comment says "Only support one unstructured mask", which reads
as unimplemented rather than impossible. Ours is not the place to change it
(the fix is in MaskAnalysis), but the reason recorded here should be the guard,
not a gather dim that is no longer involved.

parseAnd's four merge arms on one 2-D tile: full vs partial structured, two unstructured on one dim, and the two-dim refusal.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
NM = 128
BM = 8
BN = 16
NVAL = 12        # the runtime `n`
MBOUND = 6       # the runtime `mbound`


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_2d_and_arms_kernel(x_ptr, ma_ptr, mb_ptr, mn_ptr, o_ptr,
                                n, mbound, BM: tl.constexpr, BN: tl.constexpr):
        om = tl.arange(0, BM)
        on = tl.arange(0, BN)
        ptrs = x_ptr + om[:, None] * BN + on[None, :]
        mm = tl.load(ma_ptr + om) != 0        # i1 on dim 0, unparseable
        m2 = tl.load(mb_ptr + om) != 0        # a SECOND i1 on dim 0
        mn = tl.load(mn_ptr + on) != 0        # i1 on dim 1

        # 1. i1 on dim 0 vs a full-extent constant on dim 0 -> pushed unchanged
        v = tl.load(ptrs, mask=mm[:, None] & (on[None, :] < n), other=-2.0)
        # 2. i1 on dim 0 vs a RUNTIME bound on dim 0 -> a cmp is synthesized
        v += tl.load(ptrs, mask=mm[:, None] & ((om[:, None] < mbound)
                                               & (on[None, :] < n)), other=-2.0)
        # 3. two i1s on dim 0 -> one arith.andi, still one unstructured mask
        v += tl.load(ptrs, mask=(mm[:, None] & m2[:, None])
                     & (on[None, :] < n), other=-2.0)
        # 4. i1 on dim 0 AND i1 on dim 1 -> refused
        v += tl.load(ptrs, mask=mm[:, None] & mn[None, :], other=-2.0)

        tl.store(o_ptr + om[:, None] * BN + on[None, :], v,
                 mask=(om[:, None] < mbound) & mn[None, :])

    return mask_2d_and_arms_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "ma": torch.randint(0, 2, (NM,), generator=g, dtype=torch.int8),
            "mb": torch.randint(0, 2, (NM,), generator=g, dtype=torch.int8),
            "mn": torch.randint(0, 2, (NM,), generator=g, dtype=torch.int8)}


def reference(inputs):
    """The four arms, summed, then stored through the store's own mask.

    This was `zeros(BM * BN)` -- a placeholder from when the kernel could not be
    compiled and its numbers were never reached. Every mask here comes from the
    inputs and both scalars are fixed, so the result is fully determined.
    """
    import torch
    x, ma, mb, mnv = inputs["x"], inputs["ma"], inputs["mb"], inputs["mn"]
    om = torch.arange(BM).reshape(BM, 1)
    on = torch.arange(BN).reshape(1, BN)
    base = x[:BM * BN].reshape(BM, BN)
    fill = torch.full((BM, BN), -2.0, dtype=torch.float32)
    mm = (ma[:BM] != 0).reshape(BM, 1)
    m2 = (mb[:BM] != 0).reshape(BM, 1)
    mn = (mnv[:BN] != 0).reshape(1, BN)
    v = torch.where(mm & (on < NVAL), base, fill)                      # 1
    v = v + torch.where(mm & ((om < MBOUND) & (on < NVAL)), base, fill)  # 2
    v = v + torch.where((mm & m2) & (on < NVAL), base, fill)           # 3
    v = v + torch.where(mm & mn, base, fill)                           # 4
    o = torch.zeros(BM, BN, dtype=torch.float32)
    smask = (om < MBOUND) & mn
    o[smask] = v[smask]
    return {"o": o.reshape(BM * BN)}


SPEC = KernelSpec(
    name="mask_2d_and_arms_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "ma_ptr": "*i8", "mb_ptr": "*i8",
               "mn_ptr": "*i8", "o_ptr": "*fp32", "n": "i32", "mbound": "i32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("ma", IN, "int8", (NM,)),
          Arg("mb", IN, "int8", (NM,)), Arg("mn", IN, "int8", (NM,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t"), ("mbound", "int32_t")],
           "scalar_values": {"n": NVAL, "mbound": MBOUND}},
    # THE CLAIMS ABOVE LIVE IN THIS ARTIFACT, so they are pinned to it: the
    # docstring drifted once already, describing gather-scatter pointers this
    # kernel had stopped producing, and no count noticed.
    expect_ir={"03-linalg.mlir": {"tts.make_tptr": 5,
                                    "tts.make_gather_scatter_tptr": 0,
                                    "tts.gather": 1, "tts.store": 1}},
)
