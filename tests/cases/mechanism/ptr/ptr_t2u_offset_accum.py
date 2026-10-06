"""How TritonToUnstructured accumulates an offset, and how it de-splats `other`.

--triton-to-unstructured replaces a pointer chain with an INTEGER chain: each
tt.addptr becomes an arith.addi on a running offset. Two things can go wrong
with that and both are handled by widening: the running offset and the new
operand can have different bit widths, and `other=` arrives as a TENSOR when
tts.gather wants a scalar.

Six loads on ONE address family (`x_ptr + sh + i`), differing only in the width
of the added offset and in the shape of `other`. That is deliberate: the width
arms are three cases of the same three-line `if` and the getScalarValue arms are
four cases of one `while` loop, and neither is legible one file at a time.

COVERS (7)
  lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp
    t2u-addptr-same-width   :338   lhsWidth == rhsWidth == 32: one arith.addi,
        no extsi anywhere.  [loads 1-4]
    t2u-addptr-extsi-lhs    :340   the ACCUMULATED offset (i32) is extended
        because this addptr's operand is i64.  [load 5]
    t2u-addptr-extsi-rhs    :346   the accumulator is already i64 and the
        INCOMING operand is i32, so the operand is extended instead.  [load 6]
    t2u-gather-other-scalarized :488  `other` present -> getScalarValue.
        [loads 2-4]
  lib/Dialect/TritonStructured/IR/TritonStructuredOps.cpp -- getScalarValue's
  four arms, one per `other=` spelling:
    gsv-const-splat-materialize :76   a dense splat constant -> a fresh SCALAR
        arith.constant.  [load 2]
    gsv-splat-walk              :89   a triton::SplatOp -> keep walking to its
        source. Only a RUNTIME scalar `other` produces one; a literal is folded
        into a dense attribute by the frontend and takes :76 instead.  [load 3]
    gsv-sitofp-truncf-rebuild   :91   an arith.sitofp (or truncf) is RECORDED
        and replayed on the scalar after the walk finishes.  [load 4]
  (gsv-scalar-passthrough, the fourth arm, needs a SCALAR pointer -- see
  tests/cases/ptr_t2u_scalar_and_roots.py.)

WHAT PROVES IT (02-unstructured.mlir)
  Loads 1-4, the i32 chain -- one addi, no extsi, and four different `default`
  operands on the same offset:
      %6  = arith.addi %5, %2 : tensor<128xi32>
      %7  = tts.gather %arg0[%6] mask = %4
            : (<f32>, tensor<128xi32>) -> tensor<128xf32>       (no default)
      %8  = tts.gather %arg0[%6] mask = %4 default = %cst       (%cst = -1.0)
      %9  = tts.gather %arg0[%6] mask = %4 default = %arg5      (the raw
                                                                 runtime scalar)
      %10 = arith.sitofp %arg6 : i32 to f32
      %11 = tts.gather %arg0[%6] mask = %4 default = %10
  Load 3's `default = %arg5` is gsv-splat-walk in one token: the tt.splat the
  frontend emitted to make `other` a tensor is stepped THROUGH, not consumed,
  and is left dead. Load 4's %10 is a scalar arith.sitofp rebuilt beside the
  gather -- the tensor-level sitofp still exists and is also dead.
  Load 5, the accumulator widened:
      %12 = arith.extsi %6 : tensor<128xi32> to tensor<128xi64>
      %13 = arith.extsi %2 : tensor<128xi32> to tensor<128xi64>   (the .to())
      %14 = arith.addi %12, %13 : tensor<128xi64>
      %15 = tts.gather %arg0[%14] : (<f32>, tensor<128xi64>) -> ...
  Load 6, the operand widened instead -- note which side the extsi is on:
      %16 = arith.extsi %3 : tensor<128xi32> to tensor<128xi64>   (j, i32)
      %17 = arith.addi %14, %16 : tensor<128xi64>
  The gather offset type is tensor<128xi32> for loads 1-4 and tensor<128xi64>
  for 5-6, which is the whole visible consequence.

WHY IT MATTERS
  `i.to(tl.int64)` looks like a no-op on the Python side and doubles the width
  of every offset op downstream, including the ones that were already emitted --
  :340 exists precisely because the accumulator was built before anyone knew.
  And the four `other=` spellings all mean "fill with a number", but only one of
  them (a literal) costs nothing.

torch equivalent: compile-time probe (`reference` returns zeros). Every load
reads the same addresses; the file is about the offset TYPE, not the values,
and the runtime `fill`/`fi` scalars are 0.

Offset-width widening in TritonToUnstructured plus getScalarValue's tensor-to-scalar arms for `other=`.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
NI = 128
BLOCK = 128
NVAL = 64        # the runtime `n`; also the bound make_inputs draws under


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_t2u_offset_accum_kernel(x_ptr, i_ptr, j_ptr, o_ptr, n, sh, fill,
                                    fi, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(i_ptr + r)
        j = tl.load(j_ptr + r)
        dm = i < n
        base = x_ptr + sh              # a scalar term, folded into the offset
        up = base + i                  # i32 accumulator + i32 offset
        up64 = base + i.to(tl.int64)   # accumulator must widen to i64
        up64b = up64 + j               # ... and now the OPERAND must widen

        a = tl.load(up, mask=dm)                       # no other
        b = tl.load(up, mask=dm, other=-1.0)           # dense splat literal
        c = tl.load(up, mask=dm, other=fill)           # runtime scalar -> splat
        d = tl.load(up, mask=dm,
                    other=tl.full((BLOCK,), fi, tl.int32).to(tl.float32))
        e = tl.load(up64, mask=dm)
        f = tl.load(up64b, mask=dm)

        tl.store(o_ptr + i, a + b + c + d + e + f, mask=dm)

    return ptr_t2u_offset_accum_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            # DISTINCT. The six loads still alias each other -- they all read
            # `base + i` -- which is what this file is here for. But the STORE
            # is `o_ptr + i`, and drawing 128 lanes out of 0..63 made several
            # write the same address with DIFFERENT values (f reads x[i + j]),
            # so the result depended on store order and no reference could be
            # written for it. A permutation keeps every IR path and drops only
            # the write conflict.
            "i": torch.randperm(NI, generator=g).to(torch.int32),
            "j": torch.randint(0, 64, (NI,), generator=g, dtype=torch.int32)}


def reference(inputs):
    """5*x[i] + x[i + j] at o[i], for the lanes `i < n` keeps.

    a..e all read `base + i` (sh is 0) and differ only in how `other=` is
    spelled, which never shows because the store is masked by the same `dm`.
    f reads `base + i + j`, the operand-widening arm.
    """
    import torch
    x, i, j = inputs["x"], inputs["i"].long(), inputs["j"].long()
    o = torch.zeros(NX, dtype=torch.float32)
    keep = i < NVAL
    o[i[keep]] = 5.0 * x[i[keep]] + x[i[keep] + j[keep]]
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_t2u_offset_accum_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "j_ptr": "*i32",
               "o_ptr": "*fp32", "n": "i32", "sh": "i32", "fill": "fp32",
               "fi": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (NI,)),
          Arg("j", IN, "int32", (NI,)), Arg("o", OUT, "float32", (NX,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t"), ("sh", "int32_t"),
                           ("fill", "float"), ("fi", "int32_t")],
           "scalar_values": {"n": NVAL, "sh": 0, "fill": 0.0, "fi": 0}},
)
