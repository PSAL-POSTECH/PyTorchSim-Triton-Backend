"""Four rank-1 masks MaskAnalysis refuses, and two LOWER BOUNDS it now places.

Split out of the old tests/cases/ptr_structured.py section C. All six loads
share the same pointer (`x_ptr + r`, as plain as it gets) and differ only in
the mask expression, which is the point -- put side by side they show exactly
where the line is. A refused mask means rewriteLoadOp returns failure, the
tt.load is left alone, and TritonToUnstructured later turns the whole thing
into a tts.gather that carries the mask as a real i1 tensor.

Loads 1 and 2 were on that list until MaskState grew a per-axis `lows`. They
are here as the POSITIVE half now, because the line they mark is the same one:
where a comparison stops being expressible as a box.

COVERS (8)   source root /workspace/triton_shared/, worktree lines
             (MaskAnalysis.cpp is lib/Analysis/MaskAnalysis.cpp)

  A. lower bounds -- REFUSED BEFORE, PLACED NOW:
  cmp-sge-nonzero            Load 1, `r >= 4`
      static_mask_dims = [28], static_mask_lows = [4]: the box is [4, 32) of a
      32-lane tile. It used to be refused because dims is a single number
      counting from the front, which can say "the first 28" and not "from the
      4th".
  cmp-sgt                    Load 2, `r > 4`
      static_mask_dims = [27], static_mask_lows = [5]. `sgt` joined the
      predicate whitelist with the lows, and not before: adding it alone would
      have landed in the sge arm, whose fold treats the mask as always-true.

  B. operand parse failures -- the predicate is fine, something below it is not:
  dispatch-unsupported-fail  MaskAnalysis.cpp:56
      parse()'s final else: the defining op is none of constant/addi/andi/cmpi/
      make_range/broadcast/splat/expand_dims/extsi. Here it is an arith.muli
      (`r * 2`), in loads 3 and 4.
  and-nonbool-operand-fail   MaskAnalysis.cpp:365
      parseAnd with failed(parse(lhs)) and !isBoolOp. `(r * 2) & 15` is an
      arith.andi on i32, not on i1, so there is no bool-recovery path and the
      failure is fatal.  Load 3.
  add-operand-parse-fail     MaskAnalysis.cpp:344
      parseAdd: one of the addi operands failed to parse. parseAdd has no
      recovery path of any kind.  Load 4 (`r * 2 + pid`).
  add-neither-scalar-fail    MaskAnalysis.cpp:209
      addStates: BOTH operands parsed fine but neither is a scalar -- two
      ranges added together has no (start, end, dim) form.  Load 5.
      remark: "Unsupported scenario where neither lhs nor rhs is a scalar".
  cmp-lhs-parse-fail         MaskAnalysis.cpp:513
      the enclosing parseCmp giving up because its LHS failed. Loads 3, 4 and 5
      all end here; masks[cmpOpDim] is already set at this point, which is what
      lets an enclosing parseAnd recover it into an unstructured mask (see
      mask_1d_bool_tensor_and.py). Nothing encloses it here, so the load simply
      fails.
  cmp-rhs-parse-fail         MaskAnalysis.cpp:517
      the same one operand later: the LHS parsed, the RHS did not. Load 6's
      bound comes out of memory (`r < bnd`), which is the ordinary shape of a
      data-dependent bound.

WHAT PROVES IT FIRED
  01-structured.mlir contains FOUR surviving tt.load ops whose pointer operand
  is a tt.addptr -- the loads were never rewritten -- next to the tts.make_tptr
  that was still created for the address:
      %10 = tt.load %4, %9 : tensor<32x!tt.ptr<f32>>
  and with tools/lower_steps.py --debug each one prints its own remark from the
  list above, followed by
      remark: PtrAnalysis: Failed to rewrite LoadOp
  02-unstructured.mlir then shows what the fallback does with the mask -- it
  becomes a real predicated gather rather than a size clamp:
      %x = "tts.gather"(%arg0, %offsets, %mask, %other) ...
  The single unmasked tts.store at the end is the control: that one WAS
  rewritten, so a missing tts.load is a decision and not a broken pipeline.

torch equivalent: six masked gathers, see reference(). Runtime scalar n=20.

NEGATIVE: six rank-1 masks MaskAnalysis must refuse. The surviving tt.load and the predicated tts.gather are the observable. Runtime i32 scalar n=20. Numerically runnable through the gather path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 32
NVAL = 20
BNDVAL = 12


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_1d_parse_reject_kernel(x_ptr, b_ptr, o_ptr, n,
                                    BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        mp = x_ptr + r

        # A. predicate whitelist
        v = tl.load(mp, mask=r >= 4, other=0.0)                 # 1 sge, rhs != 0
        v += tl.load(mp, mask=r > 4, other=0.0)                 # 2 sgt

        # B. operands under a legal slt
        v += tl.load(mp, mask=((r * 2) & 15) < n, other=0.0)    # 3 i32 andi
        v += tl.load(mp, mask=(r * 2 + pid) < n, other=0.0)     # 4 addi of a muli
        rr = r + r
        v += tl.load(mp, mask=rr < n, other=0.0)                # 5 two ranges
        bnd = tl.load(b_ptr + r)
        v += tl.load(mp, mask=r < bnd, other=0.0)               # 6 bound from memory

        tl.store(o_ptr + r, v)

    return mask_1d_parse_reject_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "b": torch.full((BLOCK,), BNDVAL, dtype=torch.int32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    r = torch.arange(BLOCK)
    base = x[0:BLOCK]
    z = torch.zeros(BLOCK, dtype=torch.float32)
    v = torch.where(r >= 4, base, z)
    v = v + torch.where(r > 4, base, z)
    v = v + torch.where(((r * 2) & 15) < NVAL, base, z)
    v = v + torch.where(r * 2 < NVAL, base, z)      # pid == 0
    v = v + torch.where(r + r < NVAL, base, z)
    v = v + torch.where(r < BNDVAL, base, z)
    return {"o": v}


SPEC = KernelSpec(
    name="mask_1d_parse_reject_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*i32", "o_ptr": "*fp32", "n": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("b", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"03-ttshared.mlir": {"tts.gather": 4}},
    extra={"scalar_values": {"n": NVAL}},
)
