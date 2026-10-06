"""A store through a repeating row index, under a mask no axis can express.

    row  = rm % VALID                     rows VALID..BM-1 alias 0..BM-VALID-1
    keep = ((rm + rn) % 7) != 0           a mask that is NOT a per-axis box
    o[row[rm], rn] = x[rm, rn]            under keep, last write winning

WHAT THIS IS FOR. A store's mask says DO NOT WRITE and the descriptor has no
per-element write-enable, so the effect has to be expressed in the one field
that is per element: the address. _empty_read_mask_carriers used to do it by reading the
destination through the store's own pointer and writing it back where the mask
was off, and argued that this is a no-op:

    THE READ USES THE SAME POINTER AS THE WRITE, so every position it touches is
    one the store was going to touch anyway -- no new addresses, and each masked
    element is handed back exactly its own value rather than a NEIGHBOUR'S.

The last clause holds only while the gathered index is INJECTIVE. `rm % VALID`
is not: two positions name one DRAM row, so a masked-off one hands back what was
there before the store and undoes the live one that had just written it.

That is the same reasoning error the scatter path had, where it was measured on
resnet18's sixteenth conv -- see tile_masked_store_short_tile.py. Both paths now
end in convert_accesses_to_transfer._park_masked_offsets, which points the masked-off
positions at the buffer's padding instead: live elements address the buffer,
parked ones address the padding, and the two sets cannot meet. This kernel is
what says the tile-pointer path does not still have the old error.

THE MASK HAS TO BE UNSTRUCTURED or this rewrite never runs: a per-axis bound
goes into static_mask_dims and the descriptor clamps it without help. `(rm + rn)
% 7` spans both axes, which is what makes it upstream's "unstructured mask" and
sends it here.

Asks whether the gather-tile-pointer store still reads the destination back, which is only a no-op when the row index is injective.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

VALID, BM, N = 48, 64, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def unstructured_mask_repeating_row_kernel(x_ptr, o_ptr,
                                               VALID: tl.constexpr,
                                               BM: tl.constexpr,
                                               N: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, N)
        row = rm % VALID
        keep = ((rm[:, None] + rn[None, :]) % 7) != 0
        v = tl.load(x_ptr + rm[:, None] * N + rn[None, :],
                    mask=(rm < VALID)[:, None], other=0.0)
        tl.store(o_ptr + row[:, None] * N + rn[None, :], v, mask=keep)

    return unstructured_mask_repeating_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(VALID * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(VALID, N)
    o = torch.zeros(VALID, N, dtype=torch.float32)
    # Programs run their elements in increasing order and rows VALID..BM-1 land
    # on rows 0..BM-VALID-1, so the later write wins. Spelled out rather than
    # left to a vectorised expression, the same way ptr_store_wrap does.
    for m in range(BM):
        loaded = x[m] if m < VALID else torch.zeros(N)
        for n in range(N):
            if (m + n) % 7 != 0:
                o[m % VALID, n] = loaded[n]
    return {"o": o.reshape(-1)}


SPEC = KernelSpec(
    name="unstructured_mask_repeating_row_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "VALID": "constexpr",
               "BM": "constexpr", "N": "constexpr"},
    constexprs={"VALID": VALID, "BM": BM, "N": N},
    args=[Arg("x", IN, "float32", (VALID * N,)),
          Arg("o", OUT, "float32", (VALID * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
