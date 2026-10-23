"""The MASKED TAIL of a row block computes an address past the end of the buffer.

`xnumel` is 48 and `XBLOCK` is 128, so 80 of the 128 rows a program covers are
mask. That is ordinary -- every tiled kernel has a tail -- and what makes this
one different is that the row index is DECOMPOSED before it addresses:

    x0 = xindex % 4
    x4 = xindex // 4
    tl.load(in_ptr0 + (x0 + 4*r + 512*x4), r_mask & xmask, other=0.0)

`// 4` multiplies the excursion by the stride it feeds. The real rows reach
`512 * (47 // 4)` = 5632; the masked ones reach `512 * (127 // 4)` = 15872,
nearly three times past the buffer's last element. The mask says not to READ
there, and the load is masked -- but the tile is STAGED by a DMA whose
descriptor is built from the address arithmetic, not from the mask, so the
transfer walks the whole range.

MEASURED ON THE MODEL, which is why this file exists:

    convnextv2, triton_npu_fused_convolution_native_layer_norm_permute_40
    User load segfault @ 0x00000000d1010000

`spad_vaddr` was 0xd0000000 and the scratchpad is `spad_size * lanes` bytes, so
that address is PAST THE WHOLE REGION -- and the kernel is not big: 7040
bytes/lane of 131072, nineteen globals. It is not a budget failure with a
misleading symptom; it is an address the tile never needed.

WHAT THIS PINS. That a masked tail cannot be read, whatever the mask arithmetic
does to the address. Reduced from the model's own kernel by keeping the two
things that make it: a row extent that is not a multiple of the block, and a
`//` on the row index feeding a stride large enough for the tail to leave the
buffer. Everything else -- the Welford, the second input, the three outputs --
is dropped, and it still segfaults.

WHY THE SHAPE IS THIS SHAPE
  * xnumel 48 against XBLOCK 128 leaves 80 mask rows, so the tail is most of the
    block rather than a few rows. A tail of one or two lands inside the buffer's
    slack and shows nothing.
  * ROWS * 512 is the buffer, and 512 * (127 // 4) exceeds it by 2.6x. Anything
    smaller may still fault, but only by landing in another buffer -- which is
    silent corruption rather than a signal.
  * the reduction is `tl.sum`, the cheapest thing that forces the row to be
    staged at all. The fault is in the staging, not the fold.

WHAT WOULD BREAK THIS: clamping the tail to the buffer rather than to `xnumel`,
which would pass this kernel and still walk off a buffer whose extent is not a
multiple of the block.

A masked row tail whose address arithmetic leaves the buffer. Reduced from convnextv2's kernel 40, which segfaults in Spike at an address past the whole scratchpad region while claiming 7040 bytes/lane.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

XNUMEL = 48          # rows that exist
XBLOCK = 128         # rows the program covers
R = 128              # the reduced extent
GROUP = 4            # what the row index is decomposed by
ROWS = XNUMEL // GROUP
NUMEL = GROUP + GROUP * (R - 1) + 512 * (ROWS - 1) + 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def masked_tail_row_address(x_ptr, o_ptr, XNUMEL: tl.constexpr,
                                XBLOCK: tl.constexpr, R: tl.constexpr,
                                GROUP: tl.constexpr):
        xindex = tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < XNUMEL
        r = tl.arange(0, R)[None, :]
        # THE DECOMPOSITION. `// GROUP` is what turns a row past the end into an
        # address far past the end: the tail rows reach 512 * (XBLOCK-1)//GROUP
        # while the real ones stop at 512 * (XNUMEL-1)//GROUP.
        x0 = xindex % GROUP
        x4 = xindex // GROUP
        v = tl.load(x_ptr + (x0 + GROUP * r + 512 * x4), xmask, other=0.0)
        tl.store(o_ptr + xindex, tl.sum(v, axis=1)[:, None], xmask)

    return masked_tail_row_address


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NUMEL, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    out = torch.zeros(XBLOCK, dtype=torch.float32)
    for i in range(XNUMEL):
        base = (i % GROUP) + 512 * (i // GROUP)
        out[i] = x[base:base + GROUP * R:GROUP].sum()
    return {"o": out}


SPEC = KernelSpec(
    name="masked_tail_row_address", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "XNUMEL": "constexpr",
               "XBLOCK": "constexpr", "R": "constexpr", "GROUP": "constexpr"},
    constexprs={"XNUMEL": XNUMEL, "XBLOCK": XBLOCK, "R": R, "GROUP": GROUP},
    args=[Arg("x", IN, "float32", (NUMEL,)),
          Arg("o", OUT, "float32", (XBLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
