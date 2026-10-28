import triton
import triton.language as tl
@triton.jit
def _topk_kernel(x_ptr, v_ptr, i_ptr, rows, E: tl.constexpr, EP: tl.constexpr, K: tl.constexpr,
                 KP: tl.constexpr, XBLOCK: tl.constexpr, LARGEST: tl.constexpr):
    r = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    c = tl.arange(0, EP)[None, :]
    j = tl.arange(0, KP)[None, :]
    live = r < rows
    if LARGEST:
        spent = float("-inf")
    else:
        spent = float("inf")
    x = tl.load(x_ptr + r * E + c, mask=live & (c < E), other=spent)
    vals = tl.zeros([XBLOCK, KP], dtype=x.dtype)
    idxs = tl.zeros([XBLOCK, KP], dtype=tl.int32)
    for n in tl.static_range(K):
        if LARGEST:
            best = tl.max(x, axis=1)[:, None]
        else:
            best = tl.min(x, axis=1)[:, None]
        at = tl.min(tl.where(x == best, c, EP), axis=1)[:, None]
        vals = tl.where(j == n, best, vals)
        idxs = tl.where(j == n, at, idxs)
        x = tl.where(c == at, spent, x)
    out = live & (j < K)
    tl.store(v_ptr + r * K + j, vals, mask=out)
    tl.store(i_ptr + r * K + j, idxs.to(tl.int64), mask=out)
