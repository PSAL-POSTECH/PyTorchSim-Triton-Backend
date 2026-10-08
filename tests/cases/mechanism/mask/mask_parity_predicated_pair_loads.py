"""Two loads masked by COMPLEMENTARY parity predicates, feeding one select.

    even lanes  load the partner at 2*(c//2)+1, negate it
    odd  lanes  load the partner at 2*(c//2)
    both        `other=0.0`, then two wheres pick which one this lane took

WHAT IT PINS. This is how Inductor spells interleaved rotary, and it is NOT the
same as loading both and selecting -- the predicate is on the LOAD:

    tmp17 = (x % 2) < 1
    tmp29 = (x % 2) >= 1
    tmp18 = tl.load(p + (449 + 2*(x1//2) + ...), tmp17 & xmask, other=0.0)
    tmp32 = tl.load(p + (448 + 2*(x1//2) + ...), tmp29 & xmask, other=0.0)
    tmp42 = tl.where(tmp17, -tmp25, tmp41)

Each lane reads ONE of the two addresses; the other load is masked off for it.
A backend that lets a masked-off lane keep the value it would have loaded, or
that widens the predicate to the whole vector, produces values of the right
magnitude built from the wrong partner -- which is what DeepSeek-V4 shows.

WHERE IT CAME FROM, and it PASSES -- which is the point. This was the fifth and
last attempt to reproduce DeepSeek-V4-Flash-Base's divergence from the kernel
source, after the pair arithmetic, `2*(x//2)` addressing, a transposed head axis
and an unanalysable mask side had each failed to. It failed too, and that is
what ended the guessing: the real cause was found by reducing the ACTUAL kernel
instead, and it was an address, not a predicate -- a constant base dropped off a
split axis in the linalg lowering's rebuildAsGatherScatter, pinned by
../ptr/ptr_const_offset_with_split_axis.py.

It stays because predication on the LOAD is a real Inductor spelling and nothing
else in the suite has complementary predicates feeding one select.

WHAT BREAKS IT AGAIN: loading both partners unconditionally and selecting after
-- the same arithmetic, none of the predication.

Interleaved rotary with the partner loaded under a PARITY predicate rather than selected after the fact. DeepSeek-V4's first divergent kernel; the Triton interpreter agrees with the CPU golden and Spike does not.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, HEADS, DIM = 32, 4, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_parity_predicated_pair_loads_kernel(
            x_ptr, cos_ptr, sin_ptr, out_ptr,
            ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr):
        y = tl.arange(0, ROWS)[:, None]
        xi = tl.arange(0, HEADS * DIM)[None, :]
        col = xi % DIM
        head = xi // DIM

        cos = tl.load(cos_ptr + (y + ROWS * (col // 2)))
        sin = tl.load(sin_ptr + (y + ROWS * (col // 2)))

        # The lane's own element, unmasked.
        self_v = tl.load(x_ptr + (col + DIM * head + HEADS * DIM * y))

        # The PARTNER, and which address it lives at depends on this lane's
        # parity -- so each load is masked to the lanes it belongs to.
        parity = xi % 2
        is_even = parity < 1
        is_odd = parity >= 1

        odd_v = tl.load(x_ptr + (1 + 2 * (col // 2) + DIM * head + HEADS * DIM * y),
                        is_even, other=0.0)
        even_v = tl.load(x_ptr + (2 * (col // 2) + DIM * head + HEADS * DIM * y),
                         is_odd, other=0.0)

        neg = tl.where(is_even, -odd_v, tl.zeros(odd_v.shape, odd_v.dtype))
        pos = tl.where(is_odd, even_v, tl.zeros(even_v.shape, even_v.dtype))
        partner = tl.where(is_even, neg, pos)

        tl.store(out_ptr + (col + DIM * y + ROWS * DIM * head),
                 self_v * cos + partner * sin)

    return mask_parity_predicated_pair_loads_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS, HEADS, DIM, generator=g, dtype=torch.float32),
            "cos": torch.randn(DIM // 2, ROWS, generator=g, dtype=torch.float32),
            "sin": torch.randn(DIM // 2, ROWS, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x, c, s = inputs["x"], inputs["cos"], inputs["sin"]
    cos = c.t()[:, None, :].expand(ROWS, HEADS, DIM // 2)   # [row, head, dim/2]
    sin = s.t()[:, None, :].expand(ROWS, HEADS, DIM // 2)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rot = torch.empty_like(x)
    rot[..., 0::2] = even * cos - odd * sin
    rot[..., 1::2] = odd * cos + even * sin
    return {"out": rot.permute(1, 0, 2).contiguous()}


SPEC = KernelSpec(
    name="mask_parity_predicated_pair_loads_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "cos_ptr": "*fp32", "sin_ptr": "*fp32",
               "out_ptr": "*fp32", "ROWS": "constexpr", "HEADS": "constexpr",
               "DIM": "constexpr"},
    constexprs={"ROWS": ROWS, "HEADS": HEADS, "DIM": DIM},
    args=[Arg("x", IN, "float32", (ROWS, HEADS, DIM)),
          Arg("cos", IN, "float32", (DIM // 2, ROWS)),
          Arg("sin", IN, "float32", (DIM // 2, ROWS)),
          Arg("out", OUT, "float32", (HEADS, ROWS, DIM))],
    grid=(1,),
    reference=reference, make_inputs=make_inputs,
)
