"""Row-wise top-k WITHOUT a sort -- k passes of argmax-and-mask.

torch equivalent: x.topk(K, dim=1)  ->  (values, indices)

WHY THIS AND NOT A SORT
  A router wants the k largest of a small row and does not care about the order
  of everything else, so sorting N to read K of them is work nobody asked for.
  Take the max, record where it was, mask that slot to -inf, repeat. That is
  what dl/moe_router.py already does by hand for k=2, unrolled twice; this is
  the same shape parameterised, and it is the version every MoE with k>2 needs.

  It also keeps the whole thing inside the ONE direction this hardware can
  reduce. Row max is an in-lane reduction (reduce_row_max.py), and so is every
  pass here: dim 0 is the lane axis and nothing crosses a bank.

WHAT MAKES THIS MORE THAN K ROW-MAXES
  The INDEX. `tl.max` alone loses where the value was, and the mask for the next
  pass has to name that slot -- so each pass needs an argmax, not a max. The
  index comes out of a min-over-position of the positions where the row equals
  its max, which is also what makes ties deterministic: on a tie the LOWEST
  index wins, matching torch.topk's documented behaviour for sorted=True.

  Masking with -inf rather than dropping the slot is what keeps the tile shape
  constant across passes. A compacting version would need a different tile every
  pass, and this hardware bakes the tile into the DMA descriptor.

WHAT PROVES IT FIRED
  Two outputs, and the indices are the harder one: values could be right with
  the argmax broken (a max is a max), but indices can only be right if each
  pass masked the slot the previous pass actually took.

Top-k as k argmax-and-mask passes, the shape a MoE router uses. Values are exact -- every element is carried, never combined -- so any drift is a lowering fault rather than arithmetic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = 128, 16, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sort_topk_kernel(x_ptr, v_ptr, i_ptr, M: tl.constexpr, N: tl.constexpr,
                         K: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        cur = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        for t in tl.static_range(K):
            best = tl.max(cur, axis=1)
            # Lowest index among the slots holding the max -- ties go to the
            # left, which is what topk(sorted=True) promises.
            at = tl.min(tl.where(cur == best[:, None], rn[None, :], N), axis=1)
            tl.store(v_ptr + rm * K + t, best)
            tl.store(i_ptr + rm * K + t, at)
            cur = tl.where(rn[None, :] == at[:, None], float("-inf"), cur)

    return sort_topk_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    v, i = inputs["x"].topk(K, dim=1)
    return {"v": v, "i": i.to(torch.int32)}


SPEC = KernelSpec(
    name="sort_topk_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "v_ptr": "*fp32", "i_ptr": "*i32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "N": N, "K": K},
    args=[Arg("x", IN, "float32", (M, N)),
          Arg("v", OUT, "float32", (M, K)),
          Arg("i", OUT, "int32", (M, K))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
