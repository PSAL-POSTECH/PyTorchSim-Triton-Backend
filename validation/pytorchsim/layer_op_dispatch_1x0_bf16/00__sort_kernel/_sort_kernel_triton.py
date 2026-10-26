import triton
import triton.language as tl
@triton.jit
def _sort_kernel(x_ptr, v_ptr, i_ptr, E, L: tl.constexpr, D: tl.constexpr, LOG: tl.constexpr,
                 DESCENDING: tl.constexpr, FLOAT: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, L)[:, None]
    reg = tl.arange(0, D)[None, :]
    pos = reg * L + lane
    live = pos < E
    if FLOAT:
        got = tl.load(x_ptr + row * E + pos, mask=live, other=0.0)
        bits = tl.where(got != got, 0x7FC00000, tl.where(got == 0, 0, got.to(tl.int32, bitcast=True)))
        k64 = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.int64)
    else:
        k64 = tl.load(x_ptr + row * E + pos, mask=live, other=0).to(tl.int64)
    if DESCENDING:
        k64 = ~k64
    k64 = tl.where(live, k64, 0x7FFFFFFFFFFFFFFF)
    hi = (k64 >> 32).to(tl.int32)
    lo = k64.to(tl.int32)
    i = pos.to(tl.int32)
    for s in tl.static_range(1, LOG + 1):
        for j in tl.static_range(s):
            if (1 << (s - 1 - j)) < L:
                at = tl.broadcast_to(lane ^ (1 << (s - 1 - j)), (L, D))
                hp = tl.gather(hi, at, 0)
                lp = tl.gather(lo, at, 0)
                ip = tl.gather(i, at, 0)
            else:
                at = tl.broadcast_to(reg ^ ((1 << (s - 1 - j)) // L), (L, D))
                hp = tl.gather(hi, at, 1)
                lp = tl.gather(lo, at, 1)
                ip = tl.gather(i, at, 1)
            lower = (pos & (1 << (s - 1 - j))) == 0
            up = (pos & (1 << s)) == 0
            lou, lpu = lo.to(tl.uint32, bitcast=True), lp.to(tl.uint32, bitcast=True)
            less = (hi < hp) | ((hi == hp) & ((lou < lpu) | ((lo == lp) & (i < ip))))
            take = (less == lower) == up
            hi = tl.where(take, hi, hp)
            lo = tl.where(take, lo, lp)
            i = tl.where(take, i, ip)
    k64 = (hi.to(tl.int64) << 32) | lo.to(tl.uint32, bitcast=True).to(tl.int64)
    if DESCENDING:
        k64 = ~k64
    if FLOAT:
        bits = k64.to(tl.int32)
        val = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)
    else:
        val = k64
    tl.store(v_ptr + row * E + pos, val.to(v_ptr.dtype.element_ty), mask=live)
    tl.store(i_ptr + row * E + pos, i.to(tl.int64), mask=live)
