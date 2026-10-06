"""A ring buffer read: a window that fits in the buffer and crosses the end ONCE.

    slot = (START + i) % CACHE        i = 0..WIN-1
    span = WIN * 1 = 16  <=  CACHE = 64        the window fits
    START = 56, so 56..63 then 0..7            exactly one seam

WHY THIS SHAPE AND NOT ANOTHER. `span <= W` is what "ring buffer" MEANS -- a
window larger than the buffer would read the same slot twice, which is not
reading a cache. So a ring read has at most one seam and therefore at most TWO
contiguous runs, and that bound comes from the definition, not from arithmetic.

WHAT IT IS FOR. Measured over the 384-kernel suite, 95 modulo sites: 39 are a
short period repeated (span > W), 25 never fire, 22 are an unflattening that
also uses `//`, 2 guard a masked tail, 1 is a permutation -- and ZERO are this.
The `shape` field's own documentation calls itself "the boundary by which
addresses wraps around", and nothing in the suite used it that way, so the one
case the name describes had no kernel. This is that kernel.

WHAT IT PINS: that the two runs stay two runs. The correct lowering is two
contiguous transfers -- [56, 64) and [0, 8) -- and not a per-element gather
index, which is what every other modulo in the suite currently becomes.

The only true ring buffer in the suite: span <= W with one seam. Two contiguous runs, so two transfers -- not a gather index.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

CACHE = 64      # the ring's slot count -- the modulus
WIN = 16        # the window read in one go; WIN <= CACHE is the ring property
START = 56      # 56..71 -> one seam at 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_ring_window_one_seam_kernel(cache_ptr, o_ptr, CACHE: tl.constexpr,
                                        WIN: tl.constexpr, START: tl.constexpr):
        i = tl.arange(0, WIN)
        # The window walks off the end of the ring and resumes at slot 0. Every
        # lane is live -- no mask -- so both runs are read for their values.
        slot = (START + i) % CACHE
        v = tl.load(cache_ptr + slot)
        tl.store(o_ptr + i, v)

    return ptr_ring_window_one_seam_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"cache": torch.randn(CACHE, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(WIN)
    return {"o": inputs["cache"][(START + i) % CACHE]}


SPEC = KernelSpec(
    name="ptr_ring_window_one_seam_kernel", kernel=kernel,
    signature={"cache_ptr": "*fp32", "o_ptr": "*fp32", "CACHE": "constexpr",
               "WIN": "constexpr", "START": "constexpr"},
    constexprs={"CACHE": CACHE, "WIN": WIN, "START": START},
    args=[Arg("cache", IN, "float32", (CACHE,)),
          Arg("o", OUT, "float32", (WIN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
