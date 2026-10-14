"""One softmax over TWO score blocks of DIFFERENT lane extents, the shape every
flash kernel has and no coverage kernel had.

WHAT IT PINS. A cross-lane reduce whose source is NARROWER than the machine, run
twice at two different widths, with the two answers combined and sent back out.
`_xlu_mask_outside_the_tile` has to pad each source with the reduction's own
identity at ITS OWN extent -- 16 lanes for one block and 32 for the other -- and
a mask taken from the wrong one is a max over the spare banks' residue and a sum
over their leftovers. Nothing above the reference check says a word (rule 12).

WHY THE TWO WIDTHS MUST DIFFER. At equal widths one mask serves both, and a
lowering that computed the extent once and reused it still answers right. The
suite's other reduce kernels all have ONE reduced tile, so none of them can tell
a per-source extent from a per-kernel one.

WHY THE BLOCKS COME OUT OF A `tl.dot` AND ARE NOT LOADED. Written with plain
loads this passes on ZERO cross-lane moves: select_lane_axis is free to bank the
tile on the axis the reduce KEEPS, and an in-lane fold is the right answer when
nothing else pins it. The array is what pins it -- a matmul result is banked on
the axis carrying N, so a row reduce over N has to cross the lanes.

    m = max(rowmax(a), rowmax(b))          two extents, one answer
    l = rowsum(exp(a - m)) + rowsum(exp(b - m))
    o = l                                  the denominator IS the output

THE DENOMINATOR IS THE OUTPUT ON PURPOSE. Dividing by it would hide a wrong `l`
behind the same wrong `l` in the numerator whenever the error is a scale, which
is exactly the error a bad mask makes.

torch equivalent: the two-block flash softmax denominator, unnormalised.

Two cross-lane reduces at two different lane extents feeding one answer. The mask outside each tile is the reduction's identity at THAT source's extent.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, D, WA, WB = 32, 16, 16, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_two_blocks_of_unequal_width_kernel(q_ptr, a_ptr, b_ptr, o_ptr,
                                                  R: tl.constexpr,
                                                  D: tl.constexpr,
                                                  WA: tl.constexpr,
                                                  WB: tl.constexpr):
        rr = tl.arange(0, R)
        rd = tl.arange(0, D)
        ra = tl.arange(0, WA)
        rb = tl.arange(0, WB)
        q = tl.load(q_ptr + rr[:, None] * D + rd[None, :])        # (R, D)
        ka = tl.load(a_ptr + rd[:, None] * WA + ra[None, :])      # (D, WA)
        kb = tl.load(b_ptr + rd[:, None] * WB + rb[None, :])      # (D, WB)
        a = tl.dot(q, ka)                                          # (R, WA)
        b = tl.dot(q, kb)                                          # (R, WB)
        m = tl.maximum(tl.max(a, axis=1), tl.max(b, axis=1))      # (R,)
        la = tl.sum(tl.exp(a - m[:, None]), axis=1)
        lb = tl.sum(tl.exp(b - m[:, None]), axis=1)
        tl.store(o_ptr + rr, la + lb)

    return reduce_two_blocks_of_unequal_width_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"q": torch.randn(R, D, generator=g, dtype=torch.float32),
            "a": torch.randn(D, WA, generator=g, dtype=torch.float32),
            "b": torch.randn(D, WB, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    q = inputs["q"]
    a, b = q @ inputs["a"], q @ inputs["b"]
    m = torch.maximum(a.max(dim=1).values, b.max(dim=1).values)
    la = torch.exp(a - m[:, None]).sum(dim=1)
    lb = torch.exp(b - m[:, None]).sum(dim=1)
    return {"o": la + lb}


SPEC = KernelSpec(
    name="reduce_two_blocks_of_unequal_width_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32",
               "o_ptr": "*fp32", "R": "constexpr", "D": "constexpr",
               "WA": "constexpr", "WB": "constexpr"},
    constexprs={"R": R, "D": D, "WA": WA, "WB": WB},
    args=[Arg("q", IN, "float32", (R, D)), Arg("a", IN, "float32", (D, WA)),
          Arg("b", IN, "float32", (D, WB)), Arg("o", OUT, "float32", (R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
