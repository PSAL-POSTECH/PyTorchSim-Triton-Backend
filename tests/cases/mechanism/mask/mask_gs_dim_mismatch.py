"""Gather on dim 0, unstructured mask on dim 1 -- the POINTER cannot say it,
so the ACCESS does. The gather pointer survives.

THE RULE THIS PINS, and it is the whole reason the file exists: which slot a
mask lands in is decided by its AXIS, never by the direction of the access.

    mask axis == gathered axis    it says WHICH ROWS PARTICIPATE, which is what
                                  a partial gather IS. An address fact -> the
                                  pointer's own gather_scatter_mask, one bit per
                                  row.  mask_gs_rebuild_same_dim.py is that.
    mask axis != gathered axis    the pointer has no field that can hold it
                                  (the verifier requires rank 1, sized to the
                                  gather offset). One bit per ELEMENT -> the
                                  access's elem_mask.  THIS FILE.

Only one dimension may be non-contiguous, and the pointer's mask has to live on
that one. What used to follow from that was that ANY other axis cost the
structured pointer entirely; that is no longer true, and the difference is
visible in this kernel's own IR.

WHAT CHANGED HERE, AND IT IS A RUNG. This file was called
mask_gs_dim_mismatch_fail.py because both accesses were REFUSED --
applyUnstructuredMask returned nullptr, rewriteLoadOp/rewriteStoreOp failed, and
--triton-to-unstructured picked the raw ops up as a per-element tts.gather and
tts.scatter. The address analysis had already built a perfectly good gather
pointer, which was then left with no consumer. Measured on this kernel:

                              before      now
      tts.make_gather_scatter_tptr   1         2      the pointer is USED now
      tts.scatter                    1         0
      tts.gather                     0         0
      accesses with elem_mask        1         2

  The load never fell -- it took the access slot already. The STORE fell two
  rungs, all the way to a per-element scatter, purely because the store path
  asked "is the pointer structured?" where the load path asked "is there an
  unfoldable mask?". Asking about the axis instead makes the two ends agree and
  costs the store nothing.

WHAT WOULD BREAK IT

    deciding the carrier by direction again      the store goes back to
                                                 tts.scatter; expect_ir catches
                                                 it as scatter 1, gs_tptr 1.
    letting the pointer take an off-axis mask    the verifier stops it -- the
                                                 slot is rank 1 and sized to the
                                                 gather offset, and this mask is
                                                 16 wide on an 8-row gather.
    _empty_read_mask_carriers missing a carrier             convert_accesses_to_transfer refuses a
                                                 read whose mask survived, so
                                                 this would be a refusal rather
                                                 than a wrong answer.

torch equivalent: column j of every gathered row participates only where um[j]
is true; rows come from the index tensor, so the gather stays a gather.

The axis picks the mask's carrier, not the direction. Gather on dim 0, mask on dim 1: the pointer cannot hold it (its slot is rank 1, sized to the gather offset), so both ends carry it on the access and the gather pointer survives. The store used to fall to tts.scatter here
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
    def mask_gs_dim_mismatch_kernel(x_ptr, i_ptr, m_ptr, o_ptr,
                                    BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        i = tl.load(i_ptr + rm)                 # gather dim is 0
        um = tl.load(m_ptr + rn) < 5            # unstructured mask on dim 1
        v = tl.load(x_ptr + i[:, None] * BN + rn[None, :],
                    mask=um[None, :], other=0.0)
        tl.store(o_ptr + i[:, None] * BN + rn[None, :], v, mask=um[None, :])

    return mask_gs_dim_mismatch_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "i": torch.randint(0, N // BN, (BM,), generator=g, dtype=torch.int32),
            "m": torch.randint(0, 10, (BN,), generator=g, dtype=torch.int32)}


def reference(inputs):
    """What the kernel actually writes.

    This used to be `zeros(N)` -- a placeholder, because the kernel could not be
    compiled and the numbers were never reached. It IS well defined: `i` may
    repeat, but every lane sharing an index writes x[idx*BN + b], which depends
    only on the index and the column, so duplicates agree and the result does
    not depend on store order.
    """
    import torch
    x, i, m = inputs["x"], inputs["i"], inputs["m"]
    o = torch.zeros(N, dtype=torch.float32)
    um = m[:BN] < 5                      # the unstructured mask on dim 1
    idx = i[:BM].long()                  # the gathered rows
    for a in range(BM):
        for b in range(BN):
            if um[b]:
                o[idx[a] * BN + b] = x[idx[a] * BN + b]
    return {"o": o}


SPEC = KernelSpec(
    name="mask_gs_dim_mismatch_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "m_ptr": "*i32",
               "o_ptr": "*fp32", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (N,)), Arg("i", IN, "int32", (BM,)),
          Arg("m", IN, "int32", (BN,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"03-ttshared.mlir": {"tts.scatter": 0, "tts.gather": 0,
                                    "tts.make_gather_scatter_tptr": 2}},
)
