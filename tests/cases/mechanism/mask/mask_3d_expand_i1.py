"""expand_dims of an i1 that is ALREADY 2-D -- parseExpandDims's rank > 1 arms.

A 3-D masked load is written `mask3 & mask2[None, :, :]`, so parseExpandDims
gets an i1 source of rank 2. The rank-1 case can RECOVER from a failed sub-parse
(see tests/cases/mask_gs_from_structured.py); the rank > 1 case has four
outcomes and three of them are refusals. This file is all four, on one 2x8x128
pointer, because the discriminator is entirely in the inner mask expression and
nothing else may vary.

COVERS (4)   lib/AnalysisStructured/MaskAnalysis.cpp
  expand-bool-nd-mask-insert                MaskAnalysis.cpp:835
      the success path: srcRank > 1, so a nullptr is inserted into `masks` at
      `axis`, keeping masks index-aligned with the new rank. Without it the
      enclosing parseAnd's masks.size() == rank assertion would trip and the
      recorded mask would end up attributed to the wrong axis.  [load 1]
  expand-bool-rank-gt1-masks-mismatch-fail  MaskAnalysis.cpp:797
      parse(src) failed, src is i1 of rank 2, and masks.size() != 2 -- the
      tl.where produces an arith.select that MaskAnalysis has no arm for, so
      `masks` was never populated and there is nothing to attribute the failure
      to.  [load 2]
  expand-bool-nd-no-mask-fail               MaskAnalysis.cpp:824
      parse failed, src is i1 of rank 2, getUnstructuredMasks() is EMPTY: the
      inner `r8[:, None] < r[None, :]` is a cmp with two non-unit dims, so
      parseCmp reset cmpOpDim to -1 and recorded nothing.  [load 3]
  expand-bool-nd-multi-mask-fail            MaskAnalysis.cpp:827
      parse failed and getUnstructuredMasks().size() > 1: the inner
      `b3[:, None] & n3[None, :]` is itself a parseAnd failure (:441) that
      leaves BOTH masks populated. Only one gather dim exists, so the outer
      expand cannot pick one either.  [load 4]

  The three failures are three DIFFERENT `if`s in the same else-branch, and the
  only thing separating them is how many entries the failed sub-parse left in
  `masks` -- 0-with-no-arm, 0-with-an-arm, and 2. Splitting them into three
  files would hide exactly the thing that distinguishes them.

WHAT PROVES IT
  Load 1 is the only one that survives pass 1. In 01-structured.mlir:
      %20 = tts.make_gather_scatter_tptr %arg0 to sizes: [2, 8, 128]
            gather_scatter_dim: 1 gather_scatter_offset: %19
            gather_scatter_mask: %10, strides: [%c1024, %c128, 1],
            offsets: [0, 0, 0]
      %21 = "tts.load"(%20, %14, %18, %cst)
            <{static_mask_dims = array<i64: -9223372036854775808, 0,
                                            -9223372036854775808>}>
    gather_scatter_dim is 1, i.e. the MIDDLE axis -- the axis the inner 8-wide
    i1 was on before `[None, :, :]` shifted it. The literal 0 sits in slot 1 of
    static_mask_dims for the same reason. That shift is what :835's nullptr
    insert exists to keep straight.
  Loads 2, 3 and 4 print, with --debug-only=triton-ptr-analysis:
      remark: MaskAnalysis failed
    three times, and their tt.load ops survive into 02-unstructured.mlir as
      %NN = tts.gather %arg0[%off] mask = %m default = %cst
            : (<f32>, tensor<2x8x128xi32>) -> tensor<2x8x128xf32>
    -- a 2048-element per-element gather each, versus load 1's scf.for of 8
    row copies.

WHY IT MATTERS
  Rank-2 i1 masks are what every batched/blocked kernel writes. Only the shape
  in load 1 -- the inner mask being an `&` whose unstructured half is on ONE
  axis -- keeps the structured lowering. The other three look equally ordinary
  in Python.

IT USED TO BE A COMPILE-TIME PROBE. The backend refused the whole kernel -- a
rank-3 tile whose lane axis would be the MIDDLE one -- so `reference` returned
zeros and the four loads were only ever read as IR. The base-offset work moved
that lane-axis decision and the kernel compiles and runs, so the reference below
is the real arithmetic and the numbers are now load-bearing. The MaskAnalysis
half is unchanged: load 1 still keeps the structured lowering and the other
three are still per-element gathers.

torch equivalent, per (b, m, n) over x viewed as [BB, BM, BN], with
b3 = ma[:BM] != 0, n3 = mn[:BN] != 0, and every b passing `b < mbound`:

    w  = (b3[m] and n < N)                  ? x[b,m,n] : -2.0
    w += (m < mbound and b3[m] and n < N)   ? x[b,m,n] : 0.0
    w += (m < n)                            ? x[b,m,n] : 0.0
    w += (b3[m] and n3[n])                  ? x[b,m,n] : 0.0

parseExpandDims with a rank-2 i1 source: the one success and the three MaskAnalysis refusals, distinguished only by what the sub-parse left in `masks`. Ran as a compile-time probe until the base-offset work moved the lane-axis decision that refused it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
NM = 256
BB = 2
BM = 8
BN = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_3d_expand_i1_kernel(x_ptr, ma_ptr, mn_ptr, o_ptr, n, mbound,
                                 BB: tl.constexpr, BM: tl.constexpr,
                                 BN: tl.constexpr):
        rb = tl.arange(0, BB)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        p3 = (x_ptr + rb[:, None, None] * (BM * BN)
              + rm[None, :, None] * BN + rn[None, None, :])
        b3 = tl.load(ma_ptr + rm) != 0          # i1 on the middle axis
        n3 = tl.load(mn_ptr + rn) != 0          # i1 on the last axis
        mb3 = rb[:, None, None] < mbound        # structured, outer axis

        # 1. inner i1 = (unstructured on axis 0) & (structured on axis 1)
        w = tl.load(p3, mask=mb3 & (b3[None, :, None]
                                    & (rn[None, None, :] < n)), other=-2.0)
        # 2. inner i1 built by tl.where -> arith.select, no MaskAnalysis arm
        mu2 = tl.where(rm[:, None] < mbound, b3[:, None] & (rn[None, :] < n),
                       (rm[:, None] < mbound) & (rn[None, :] < n))
        w += tl.load(p3, mask=mb3 & mu2[None, :, :], other=0.0)
        # 3. inner i1 = a cmp across two non-unit axes -> nothing recorded
        m2a = rm[:, None] < rn[None, :]
        w += tl.load(p3, mask=mb3 & m2a[None, :, :], other=0.0)
        # 4. inner i1 = two unstructured masks on two axes -> two recorded
        m2b = b3[:, None] & n3[None, :]
        w += tl.load(p3, mask=mb3 & m2b[None, :, :], other=0.0)

        tl.store(o_ptr + rb[:, None, None] * (BM * BN)
                 + rm[None, :, None] * BN + rn[None, None, :], w)

    return mask_3d_expand_i1_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "ma": torch.randint(0, 2, (NM,), generator=g, dtype=torch.int8),
            "mn": torch.randint(0, 2, (NM,), generator=g, dtype=torch.int8)}


def reference(inputs):
    import torch
    n, mbound = 100, 2
    x = inputs["x"][:BB * BM * BN].reshape(BB, BM, BN)
    b3 = inputs["ma"][:BM] != 0
    n3 = inputs["mn"][:BN] != 0
    rb = torch.arange(BB)[:, None, None]
    rm = torch.arange(BM)[:, None]
    rn = torch.arange(BN)[None, :]
    mb3 = rb < mbound
    zero = torch.zeros_like(x)

    m1 = b3[:, None] & (rn < n)
    w = torch.where(mb3 & m1[None], x, torch.full_like(x, -2.0))
    mu2 = torch.where(rm < mbound, b3[:, None] & (rn < n),
                      (rm < mbound) & (rn < n))
    w = w + torch.where(mb3 & mu2[None], x, zero)
    w = w + torch.where(mb3 & (rm < rn)[None], x, zero)
    w = w + torch.where(mb3 & (b3[:, None] & n3[None, :])[None], x, zero)
    return {"o": w.reshape(-1)}


SPEC = KernelSpec(
    name="mask_3d_expand_i1_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "ma_ptr": "*i8", "mn_ptr": "*i8",
               "o_ptr": "*fp32", "n": "i32", "mbound": "i32",
               "BB": "constexpr", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BB": BB, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("ma", IN, "int8", (NM,)),
          Arg("mn", IN, "int8", (NM,)),
          Arg("o", OUT, "float32", (BB * BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # The [2] mask's broadcast takes the default lane axis, of 2, and the `&` spreads it; each
    # [8,128] broadcast reaches it as one 128-round gather, none through DRAM.
    expect_ir={"04-transformed.mlir": {"dram_arg = -1": 0}},
    extra={"scalar_args": [("n", "int32_t"), ("mbound", "int32_t")],
           "scalar_values": {"n": 100, "mbound": 2}},
)
