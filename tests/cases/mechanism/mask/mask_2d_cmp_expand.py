"""How MaskAnalysis PARSES one mask: cmp, expand_dims and broadcast.

`mask=(om[:, None] < m) & (on[None, :] < n)` is four ops in the TTIR, and
MaskState::parse walks all of them: tt.expand_dims, tt.broadcast, arith.cmpi.
This file is the arms of those three parsers -- what each one writes into
`dims` (the per-axis size clamp) and into `masks` (the per-axis i1 tensor), and
the two shapes that make parsing fail outright.

SIBLING: tests/cases/mask_2d_and_arms.py, which consumes what this file
produces -- it is about merging two parsed masks. Same tile shape, so the two
kernels' IR lines up line for line.

COVERS (10)   lib/AnalysisStructured/MaskAnalysis.cpp
  dispatch-expanddims        MaskAnalysis.cpp:50
  dispatch-broadcast         MaskAnalysis.cpp:46
      the two TypeSwitch arms of MaskState::parse that make a 1-D comparison
      into a 2-D mask at all.  [all loads]
  expand-dims-insert-one     MaskAnalysis.cpp:841
      parseExpandDims's unconditional tail: a constant 1 is inserted into dims
      at `axis`. That 1 is what parseCmp's dim scan skips and what
      parseBroadcast then widens.  [loads 1, 2, 4]
  broadcast-widen-dim        MaskAnalysis.cpp:741
      srcShape[i] < dstShape[i] -> dims[i] = the FULL destination extent, as a
      compile-time IndexAttr. This is the constant that lets parseAnd take its
      "full mask" shortcut.  [loads 1, 2, 4]
  broadcast-dim-equal        MaskAnalysis.cpp:739
      srcShape[i] == dstShape[i] -> dims[i] left alone, so a previously
      narrowed dim (min(mbound, 8)) SURVIVES the broadcast instead of being
      reset to 8.  [loads 1, 2, 4]
  expand-bool-1d-success     MaskAnalysis.cpp:818
      isBoolOp && rank-1 src && parse(src) SUCCEEDED: masks is resized to 2 and
      BOTH entries forced to nullptr, then a 1 is inserted into dims. The
      difference from load 1 is where the `[:, None]` sits relative to the
      `<`.  [load 2]
  cmp-multi-nonunit-shape    MaskAnalysis.cpp:466
      the cmp RESULT has two dims of extent != 1, so cmpOpDim is reset to -1
      and the scan breaks -- masks stay all-null, so nothing can be recovered
      later.  [load 3]
  cmp-multi-nonunit-lhsdim-reject  MaskAnalysis.cpp:536
      the follow-up check on lhsState.dims: more than one entry is not the
      constant 1 -> "more than one dimension with size larger than 1".  [load 3]
  cmp-nd-collapse-mask-record      MaskAnalysis.cpp:486
      cmpOpDim != -1 and rank > 1 -> a tensor.collapse_shape to 1-D is emitted
      EAGERLY and recorded as masks[cmpOpDim], before anyone knows whether the
      parse will succeed. On the success path it is dead IR.  [load 4]
  expand-nonbool-src-fail    MaskAnalysis.cpp:795
      parse(src) failed and the src element type is not i1: `midx[:, None]`
      expands an i32 tensor that came out of memory, so there is no recovery
      path (contrast expand-bool-1d-recover, tests/cases/
      mask_gs_from_structured.py).  [load 4]
  broadcast-src-parse-fail   MaskAnalysis.cpp:735
      the enclosing parseBroadcast then fails too -- but it propagates `masks`
      populated by the nested parse, which is how load 4 still ends up with a
      usable per-element mask.  [load 4]

WHAT PROVES IT
  Loads 1 and 2 (`(om[:, None] < mbound) & (on[None, :] < n)` and its
  `(om < mbound)[:, None]` spelling) are the fully structured baseline: the
  whole module contains ZERO tts.make_gather_scatter_tptr, just size clamps on
  the tts.load --
      %24 = arith.minsi %c8, %23 : index          (dims[0] = min(mbound, 8))
      %27 = arith.minsi %c16_2, %26 : index       (dims[1] = min(n, 16))
      %29 = arith.minsi %25, %c8_4 : index        (the broadcast, :739, keeping
                                                   the narrowed dim)
      %31 = "tts.load"(%10, %29, %30, %cst_6)
            <{static_mask_dims = array<i64: -9223372036854775808,
                                            -9223372036854775808>}>
    Two dynamic dims and no mask operand at all: everything parsed.
  Loads 3 and 4 both fail and stay raw in 01-structured.mlir:
      %49 = tt.load %11, %48, %cst : tensor<8x16x!tt.ptr<f32>>   (load 3)
      %58 = tt.load %11, %54, %cst : tensor<8x16x!tt.ptr<f32>>   (load 4)
    becoming per-element tts.gather ops in 02-unstructured.mlir.
  Load 4 leaves the evidence for :486 behind even though it failed -- the
  eagerly emitted collapse is still in 01-structured.mlir with no consumer:
      %collapsed_15 = tensor.collapse_shape %52 [[0, 1]]
                      : tensor<8x1xi1> into tensor<8xi1>
    That `8x1xi1 into 8xi1` is cmp-nd-collapse-mask-record's fingerprint. (The
    `1x16xi1 into 16xi1` collapses next to it are the same branch firing on the
    `on[None, :] < n` half of every mask; they are dead too.) Nothing consumes
    any of them because the enclosing parse failed at :735 -- a mask recorded
    into `masks` only survives if the CALLER is a parseExpandDims that can
    recover, which is what tests/cases/mask_gs_from_structured.py shows.

WHY IT MATTERS
  Loads 1 and 2 are the same predicate written two ways -- `(a < b)[:, None]`
  versus `a[:, None] < b` -- and they take different arms of parseExpandDims.
  They happen to agree here, but the recovery behaviour when the sub-parse
  FAILS is completely different between the two shapes, which is why the pair
  is worth keeping side by side.

torch equivalent: compile-time probe (`reference` returns zeros); loads 3 and 4
exist to be refused.

parseCmp / parseExpandDims / parseBroadcast arms on one 2-D tile, including the two shapes that cannot be parsed at all.
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
    def mask_2d_cmp_expand_kernel(x_ptr, mi_ptr, o_ptr, n, mbound,
                                  BM: tl.constexpr, BN: tl.constexpr):
        om = tl.arange(0, BM)
        on = tl.arange(0, BN)
        ptrs = x_ptr + om[:, None] * BN + on[None, :]
        midx = tl.load(mi_ptr + om)            # i32, NOT i1

        # 1. the ordinary 2-D structured mask: expand -> broadcast -> cmp
        v = tl.load(ptrs, mask=(om[:, None] < mbound) & (on[None, :] < n),
                    other=0.0)
        # 2. the [:, None] moves OUTSIDE the compare: an i1 whose parse worked
        v += tl.load(ptrs, mask=(om < mbound)[:, None] & (on[None, :] < n),
                     other=0.0)
        # 3. a compare between two non-unit axes: no cmpOpDim, no mask recorded
        v += tl.load(ptrs, mask=om[:, None] < on[None, :], other=0.0)
        # 4. expand_dims of a loaded i32: parse fails, but the cmp had already
        #    recorded its collapsed self as the mask
        v += tl.load(ptrs, mask=(midx[:, None] < mbound) & (on[None, :] < n),
                     other=0.0)

        tl.store(o_ptr + om[:, None] * BN + on[None, :], v)

    return mask_2d_cmp_expand_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "mi": torch.randint(0, BM, (NM,), generator=g, dtype=torch.int32)}


def reference(inputs):
    """The four masked loads, summed.

    This was `zeros(BM * BN)` -- a placeholder, because the kernel could not be
    compiled and the numbers were never reached. Arms 1 and 2 are the same mask
    written two ways, so the tile they keep is added twice.
    """
    import torch
    x, mi = inputs["x"], inputs["mi"]
    om = torch.arange(BM).reshape(BM, 1)
    on = torch.arange(BN).reshape(1, BN)
    base = x[:BM * BN].reshape(BM, BN)
    z = torch.zeros(BM, BN, dtype=torch.float32)
    inb = (om < MBOUND) & (on < NVAL)
    v = torch.where(inb, base, z) * 2                              # arms 1, 2
    v = v + torch.where(om < on, base, z)                          # arm 3
    v = v + torch.where((mi[:BM].reshape(BM, 1) < MBOUND) & (on < NVAL),
                        base, z)                                   # arm 4
    return {"o": v.reshape(BM * BN)}


SPEC = KernelSpec(
    name="mask_2d_cmp_expand_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "mi_ptr": "*i32", "o_ptr": "*fp32",
               "n": "i32", "mbound": "i32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("mi", IN, "int32", (NM,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t"), ("mbound", "int32_t")],
           "scalar_values": {"n": NVAL, "mbound": MBOUND}},
)
