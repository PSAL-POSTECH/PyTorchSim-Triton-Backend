"""tts.gather / tts.scatter: the address PtrAnalysis gave up on entirely.

When the index tensor has more than one non-unit dimension there is no
structured pointer to build and no gather DIM to pick, so
--triton-to-unstructured takes over: it walks the tt.addptr chain accumulating
an INTEGER offset, deletes every pointer-typed op, and emits one tts.gather per
tt.load and one tts.scatter per tt.store, keeping the offset tensor at its full
8x16 shape. b11 then reads those two ops directly.

Five ops in one kernel, because the optional operands are what is being shown:
gather with/without mask, with/without `other`, scatter with/without mask. They
are only comparable side by side.

COVERS (19)
  The ops themselves -- lib/Dialect/TritonStructured/IR/TritonStructuredDialect.td
  (TableGen: gdb cannot break on these, the branch census counted them as
  unmeasurable; the assembly below is the evidence)
    tts-gather-emitted        .td:262    tts-scatter-emitted     .td:286
    tts-gather-offset-tensor  .td:273    tts-scatter-offset-tensor .td:297
    tts-gather-mask-absent    .td:274    tts-scatter-mask-absent .td:298
    tts-gather-mask-present   .td:274    tts-scatter-mask-present .td:298
    tts-gather-other-absent   .td:275
    tts-gather-other-present  .td:275
  Emission -- lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp
    t2u-offsettype-tensor-of-ptr  :191   the offset type mirrors the ptr tensor
                                         with an integer element type
    t2u-getbitwidth-tensor        :208   width taken from tensor<8x16xi32>
    t2u-seed-kernel-ptr-arg       :251   each !tt.ptr argument seeded with an
                                         `arith.constant 0 : i32`
    t2u-collect-ptrusers          :411   loads/stores deferred to a second loop
                                         so every offset op exists first
    t2u-addptr-same-width         :338   i32 accumulator + i32 offset, so
                                         neither extsi guard fires
    t2u-shapeop-clone-on-offsets  :374   the pointer-typed tt.splat is CLONED
                                         onto the integer offset
    t2u-tts-maketptr-base-is-kernel-arg-skip :405  the index load IS structured;
                                         its tts.make_tptr is left alone
                                         instead of having a zero offset folded
                                         into it
    t2u-emit-gather               :482    t2u-emit-scatter        :503
    t2u-gather-no-mask            :495    t2u-scatter-no-mask     :505
    t2u-gather-mask-present       :495    t2u-scatter-mask-present :505
    t2u-gather-other-absent       :487
    t2u-gather-other-scalarized   :488   getScalarValue de-splats `other`
    t2u-delete-and-rauw-offsets   :567   zero tt.addptr and zero pointer-typed
                                         tt.splat survive
    t2u-pass-success-canonicalize-cse :584  the seed constants and their
                                         `addi %c0, %off` are folded away, so
                                         the gather's offset IS the index
    gsv-const-splat-materialize   TritonStructuredOps.cpp:76   the dense<0.0>
                                         tensor `other` becomes a SCALAR
                                         arith.constant next to the gather
  Lowering -- a02_convert_tts_to_pytorchsim.py, the 3rd-rung ops
    gather-indirect-transfer   a tts.gather is ONE indirect transfer over the
        offset tensor: dram_stride is all zeros (the index carries the whole
        address) and offset_stride is 1.
    gather-mask-is-a-select    a per-element mask on a gather is not a box, so
        _empty_read_mask_carriers turns it into an unmasked read plus a select
        BEFORE b11 sees it. All three loads therefore emit the SAME transfer:
            {indirect, offset_stride = 1, dram_stride = [0, 0]}
        with and without mask, with and without `other`. That the three are
        indistinguishable here is the point -- the difference lives in the
        select above them, not in the DMA.
    scatter-unmasked           store 1: the same shape, MVOUT.
    scatter-masked-parks       store 2: a masked scatter cannot select, because
        a write that must not happen has to not happen. It carries
        pytorchsim.dram_bound and the DMA drops every address outside the argument.
        That attribute is the only thing telling the two stores apart.

WHAT PROVES IT
  02-unstructured.mlir is 13 lines long and contains all five ops, verbatim:
      %4 = tts.gather %arg0[%1] : (<f32>, tensor<8x16xi32>) -> tensor<8x16xf32>
      %5 = tts.gather %arg0[%1] mask = %3
           : (<f32>, tensor<8x16xi32>) -> tensor<8x16xf32>
      %6 = tts.gather %arg0[%1] mask = %3 default = %cst
           : (<f32>, tensor<8x16xi32>) -> tensor<8x16xf32>
      tts.scatter %4 into %arg2[%1]
           : tensor<8x16xf32> into (<f32>, tensor<8x16xi32>)
      tts.scatter %7 into %arg2[%1] mask = %3
           : tensor<8x16xf32> into (<f32>, tensor<8x16xi32>)
  Three gathers on the SAME %1 offset, differing only in the optional clauses;
  %1 is the raw index load and `%cst = arith.constant 0.000000e+00 : f32` is a
  SCALAR, de-splatted from the dense tensor the frontend built. The base is
  `%arg0` itself -- no tt.addptr and no pointer-typed tt.splat survives anywhere
  in the module, and the `arith.constant 0 : i32` seeds are gone too, folded by
  the canonicalizer the pass runs on success.
  Note the INDEX load above them is a normal tts.make_tptr/tts.load pair: the
  addresses of the indices are perfectly affine. Only the loads THROUGH those
  indices fell out.
  05-unstruct-memref.mlir, the three gather bodies:
      no mask   : ins(%4) -> arith.index_cast + tensor.extract + linalg.yield
      mask, no other:
          %13 = scf.if %in_2 -> (f32) { ... scf.yield %extracted }
                else { %cst_3 = arith.constant 0.0 : f32   <- created INSIDE
                       scf.yield %cst_3 }
      mask + other:
          the same scf.if, but the else yields %cst from OUTSIDE the region
  and the two scatter bodies -- `ins(%4, %11 : ...)` with no outs(), body
  `memref.store` versus `scf.if %m { memref.store; scf.yield }` (no else).

WHY IT MATTERS
  This is the floor of the lowering: everything else in this directory exists to
  avoid landing here. A tts.gather is one memory op per ELEMENT with no
  contiguity left to exploit, and the only thing that decides whether a kernel
  gets it is whether PtrAnalysis could name a single gather dimension.

torch equivalent: o.flat[uidx] = x.flat[uidx] for the unmasked pass, then the
masked one on top; `reference` computes it directly.

The unstructured floor: tts.gather/tts.scatter with every combination of the optional mask and other operands.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 1024
NI = 256
BM = 8
BN = 16
KEEP = 40


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_unstructured_gather_scatter_kernel(x_ptr, i_ptr, o_ptr, k,
                                               BM: tl.constexpr,
                                               BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        # an 8x16 index block: BOTH dims > 1, so no single gather dim exists
        uidx = tl.load(i_ptr + rm[:, None] * BN + rn[None, :])
        umk = uidx < k

        ua = tl.load(x_ptr + uidx)                      # no mask, no other
        ub = tl.load(x_ptr + uidx, mask=umk)            # mask, no other
        uc = tl.load(x_ptr + uidx, mask=umk, other=0.0)  # mask and other

        tl.store(o_ptr + uidx, ua)                      # scatter, no mask
        tl.store(o_ptr + uidx, ub + uc, mask=umk)       # scatter, mask

    return ptr_unstructured_gather_scatter_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # duplicate-free so the two scatters are well defined and order-independent
    idx = torch.zeros(NI, dtype=torch.int32)
    idx[:BM * BN] = torch.randperm(NX, generator=g)[:BM * BN].to(torch.int32)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": idx}


def reference(inputs):
    import torch
    x, i = inputs["x"], inputs["i"].long()[:BM * BN]
    o = torch.zeros(NX, dtype=torch.float32)
    keep = i < KEEP
    o[i] = x[i]                                   # the unmasked scatter
    masked = torch.where(keep, x[i], torch.zeros(())) * 2.0
    o[i[keep]] = masked[keep]                     # the masked one, on top
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_unstructured_gather_scatter_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "k": "i32", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (NI,)),
          Arg("o", OUT, "float32", (NX,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("k", "int32_t")], "scalar_values": {"k": KEEP}},
)
