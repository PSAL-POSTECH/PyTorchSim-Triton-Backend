"""One unstructured mask on a STRUCTURED pointer, and the pointer STAYS structured.

WHAT THE PIPELINE DOES. `um` is loaded from memory, so MaskState::parse leaves
exactly one non-null entry in `masks`: dim 1 carries it, dim 0 is clean. That is
the one shape applyUnstructuredMask can repair, and upstream's repair is to
DEMOTE the tts.make_tptr to a tts.make_gather_scatter_tptr so the mask has
somewhere to live. We do not take that repair: `unstructured-mask-as-select`
keeps the pointer, reads the masked axis WHOLE, and puts the mask on an
arith.select over the loaded tile. pipeline.py does not name it -- it is the
fork's default, and stage 3's pass list is option-free.

THE DEMOTION ADDS NOTHING. The index it synthesises is
`splat(offset/stride) + make_range(0, size)` -- the identity permutation, the
affine row the DMA would have walked anyway -- so what the demotion buys is a
slot for the mask and what it costs is an index buffer and an indirect transfer.
NEITHER OPTION EXISTS NOW -- TritonToStructured takes four and these are not
among them, because the arms they switched on are gone. What follows is what
they did while they were there.

AND THE SELECT IS NOT THERE ANY MORE EITHER. `structured-elem-mask` carries the
mask in tts.load's own `elem_mask` and leaves the realising to _empty_read_mask_carriers,
where the machine is in view -- so 03-linalg has the descriptor and the mask,
and the two selects appear one stage down. Both halves are pinned below.

And convert_accesses_to_transfer takes the mask straight back off. See the sibling files for
the two ends this sits between: mask_structured_dims.py (mask fully absorbed
into `dims`) and mask_unparsed_to_gather.py (unparseable, whole load falls to a
per-element tts.gather).

WHAT THIS PINS (expect_ir on 03-linalg.mlir)

                                  option off   option on
      tts.make_gather_scatter_tptr     2            0
      tts.make_tptr                    2            4
      arith.select (03-linalg)        0            0

  Two loads, not one, because the offset/stride OpFoldResults reach the demotion
  in two shapes -- all-static (load 1) and pid-dependent (load 2) -- and both
  have to stop demoting, not just the easy one.

WHAT MOVED, AND WHERE IT IS STILL COVERED. This file used to cover six lines of
applyUnstructuredMask (the ofrToI32Value arms at PtrAnalysis.cpp:77/81/84, the
:73 make_tptr branch, and :119's `dims[dim] = 0`). With the option on they run
for NEITHER end any more. The line here used to read "a masked STORE still
demotes, because park needs a per-position address to move and a structured
pointer has none" -- that stopped being true when convert_accesses_to_transfer learned
to build the per-position address ITSELF from the descriptor it kept, so the
store carries `elem_mask` too. Measured on the three kernels that had the
demotion: mask_2d_and_arms, mask_col_flag_broadcast_over_rows and this file all
report `tts.make_gather_scatter_tptr` 0. Those arms are upstream code and
upstream still tests them:
tests/Conversion/TritonToStructured/unstructured_mask_3d_kernel.mlir, plus this
fork's unstructured_mask_as_select.mlir for the option's own branch, and
mask_elem_on_wrapped_store.py is the store end at its hardest -- a mask and a
wraparound on one descriptor.

torch equivalent: column j of every row is read only where um[j] is true; load 2
reads at a pid-dependent column offset, so the value differs per program id.

A structured ptr with one unstructured mask keeps its make_tptr: the mask goes to a select instead of demoting the pointer. Two loads for the two offset/stride shapes.
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
    def mask_gs_from_structured_kernel(x_ptr, m_ptr, o_ptr,
                                       BM: tl.constexpr, BN: tl.constexpr):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        um = tl.load(m_ptr + rn) < 5           # unstructured mask, dim 1

        # (1) offset/stride of dim 1 are static -> Attribute -> constant -> trunci
        a = tl.load(x_ptr + rm[:, None] * BN + rn[None, :],
                    mask=um[None, :], other=0.0)
        # (2) offset of dim 1 is an index-typed SSA value -> index_cast
        b = tl.load(x_ptr + rm[:, None] * BN + (pid * BN + rn)[None, :],
                    mask=um[None, :], other=0.0)

        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], a + b)

    return mask_gs_from_structured_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "m": torch.randint(0, 10, (N,), generator=g, dtype=torch.int32)}


def reference(inputs):
    """What the kernel computes, for the LAST program id.

    This returned all-zeros while the kernel was a compile-time probe that never
    reached Spike, so nothing ever checked it. It does now, and the placeholder
    is what a working lowering would have been marked wrong against.

    THE LAST pid WINS. The store address does not depend on the program id, so
    every pid writes the same BM*BN elements and the wrapper's sequential grid
    loop leaves the last one standing. Only `b` reads a pid-dependent address
    (`pid * BN + rn`), which is exactly the splat that used to be dropped.
    """
    import torch
    pid = SPEC.grid[0] - 1
    rm, rn = torch.arange(BM), torch.arange(BN)
    um = inputs["m"][rn] < 5
    x = inputs["x"]
    zero = torch.zeros((), dtype=torch.float32)
    a = torch.where(um[None, :], x[rm[:, None] * BN + rn[None, :]], zero)
    b = torch.where(um[None, :], x[rm[:, None] * BN + (pid * BN + rn)[None, :]],
                    zero)
    return {"o": (a + b).reshape(-1)}


SPEC = KernelSpec(
    name="mask_gs_from_structured_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "m_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (N,)), Arg("m", IN, "int32", (N,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(2,), reference=reference, make_inputs=make_inputs,
    expect_ir={"03-linalg.mlir": {"tts.make_gather_scatter_tptr": 0,
                                    "tts.make_tptr": 4},
               # THE SELECT MOVED DOWN A STAGE and the claim moved with it.
               # Upstream carries the mask in tts.load's elem_mask; turning it
               # into a select is convert_accesses_to_transfer's first phase, which is
               # where the machine is in view. Both halves are pinned.
               "04-passes/03-torchsim-convert-accesses-to-transfer.mlir":
                   {"arith.select": 2}},
)
