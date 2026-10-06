"""A masked gather with a non-zero `other=` -- the clamp that also guards the index.

This was a SILENT WRONG ANSWER before: emit_gather never read the tts.load's
`static_mask_dims`, so the mask and the fill were both dropped and the kernel
returned x[idx] in the masked tail. Measured then: 32/128 elements wrong, with
no diagnostic anywhere in the pipeline.

WHY THE CLAMP IS ENOUGH -- and why it matters more here than on a dense load

  Spike applies the box clamp BEFORE the indirect lookup
  (torchsim_mvin_common.h):

      dma_buffer[buffer_idx] = in_box ? d_addr : 0;      // clamp first
      ...
      d_addr = dma_buffer[d_idx];
      is_used_vlane &= (d_addr != 0);                    // out-of-box -> dead
      if (indirect_mode && is_used_vlane) { ...read index... }
      val = is_used_vlane ? MMU.load(d_addr) : desc_fill;

  so an out-of-box position never reads the index array at all. That is the
  "gather 인덱스 보호" the README calls the one thing the DMA mask is really
  needed for: the masked tail of an index buffer holds whatever the index load
  left there, and dereferencing it would be an arbitrary DRAM read.

torch equivalent: see `reference` -- x[idx[:n]] then the fill.

Masked gather with a non-zero fill. Was silently unmasked; the DMA box clamp both fills the tail and stops the index from being read.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
NLIM = 96
FILL = -99.0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_masked_fill_kernel(x_ptr, idx_ptr, o_ptr, n,
                                      BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(idx_ptr + r)
        # unmasked store, so the fill is observable
        v = tl.load(x_ptr + i, mask=r < n, other=-99.0)
        tl.store(o_ptr + r, v)

    return ptr_gather_masked_fill_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": torch.randperm(N, generator=g)[:BLOCK].to(torch.int32)}


def reference(inputs):
    import torch
    o = torch.full((BLOCK,), FILL, dtype=torch.float32)
    o[:NLIM] = inputs["x"][inputs["idx"][:NLIM].long()]
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_gather_masked_fill_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32",
               "n": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("idx", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t")], "scalar_values": {"n": NLIM}},
)
