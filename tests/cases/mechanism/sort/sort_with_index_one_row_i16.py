"""A stable argsort of 8 int64 keys into int16 indices, one program: Inductor's sort_with_index
(MoE dispatch at T=1, issue #214), its bitonic network written out here as Inductor writes it.

A compare-and-swap step reshapes the tile, and materialize-layouts cut one of those reshapes'
destinations into digits. Nothing read that destination, so the destination was replaced by a
fresh one. The reshape that had made the old destination was left unused, but the layout
plan still turned it into a copy, a cross-lane permute of an empty tile, and a copy writes, so
nothing erased it. togcheck then refused the TOG: "tile 1 reads buffer 16, which nothing in the
work item writes". Breaks if what an unread destination leaves unused survives to the plan.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _swap(x, idxs, flip, i: tl.constexpr, n_dims: tl.constexpr):
        shape: tl.constexpr = [2**i, 2, 2 ** (n_dims - i - 1)]
        y = tl.reshape(x, shape)
        right_mask = tl.arange(0, 2)[None, :, None].to(tl.int64)
        left_mask = (1 - right_mask).to(tl.int64)
        left = tl.reshape(tl.broadcast_to(tl.sum(y * left_mask, 1)[:, None, :], shape), x.shape)
        right = tl.reshape(tl.broadcast_to(tl.sum(y * right_mask, 1)[:, None, :], shape), x.shape)
        y_idx = tl.reshape(idxs, shape)
        left_idx = tl.reshape(tl.broadcast_to(
            tl.sum(y_idx * left_mask.to(y_idx.dtype), 1)[:, None, :], shape), x.shape)
        right_idx = tl.reshape(tl.broadcast_to(
            tl.sum(y_idx * right_mask.to(y_idx.dtype), 1)[:, None, :], shape), x.shape)
        cond = (left > right) | ((left == right) & (left_idx > right_idx))
        cond = (cond ^ flip).to(tl.int1)
        ret = x ^ tl.where(cond, left ^ right, tl.zeros_like(x))
        new_idxs = idxs ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(idxs))
        return ret, new_idxs

    @triton.jit
    def _merge(x, idxs, stage: tl.constexpr, alternating: tl.constexpr, n_dims: tl.constexpr):
        if alternating:
            shape: tl.constexpr = [2 ** (n_dims - 1 - stage), 2, 2**stage]
            flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
        else:
            flip = False
        for i in tl.static_range(stage):
            x, idxs = _swap(x, idxs, flip, i + (n_dims - stage), n_dims)
        return x, idxs

    @triton.jit
    def sort_with_index_one_row_i16_kernel(x_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.load(x_ptr + r)
        idxs = r.to(tl.int16)
        n_dims: tl.constexpr = 3
        for s in tl.static_range(1, n_dims + 1):
            x, idxs = _merge(x, idxs, s, s < n_dims, n_dims)
        tl.store(o_ptr + r, idxs)

    return sort_with_index_one_row_i16_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(0, 4, (N,), generator=g).to(torch.int64)}


def reference(inputs):
    import torch
    return {"o": torch.sort(inputs["x"], stable=True).indices.to(torch.int16)}


SPEC = KernelSpec(
    name="sort_with_index_one_row_i16_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "o_ptr": "*i16", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "int64", (N,)), Arg("o", OUT, "int16", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
