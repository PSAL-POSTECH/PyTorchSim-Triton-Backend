import triton
import triton.language as tl
@triton.jit
def _grouped_mm_kernel(a_ptr, b_ptr, grp_ptr, start_ptr, rows_ptr, c_ptr,
                       sbg: tl.constexpr, sbk: tl.constexpr, sbn: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                       N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.load(grp_ptr + pid_m)
    nr = tl.load(rows_ptr + pid_m)
    rm = tl.arange(0, BM)
    rows = tl.load(start_ptr + pid_m) + rm
    live = rm < nr
    rn = pid_n * BN + tl.arange(0, BN)
    ncols = rn < N
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, tl.where(nr > 0, KP, 0), BK):
        kk = k0 + rk
        a = tl.load(a_ptr + rows[:, None] * K + kk[None, :], mask=live[:, None] & (kk < K)[None, :], other=0.0)
        b = tl.load(b_ptr + g * sbg + kk[:, None] * sbk + rn[None, :] * sbn,
                    mask=(kk < K)[:, None] & ncols[None, :], other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rows[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=live[:, None] & ncols[None, :])
