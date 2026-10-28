import triton
import triton.language as tl
@triton.jit
def _grouped_mm_kernel(a_ptr, b_ptr, group_ptr, start_ptr, rows_ptr, c_ptr,
                       sbg: tl.constexpr, sbk: tl.constexpr, sbn: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                       N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Group g, the program_id(0)-th group with rows: its rows times B[g], BM rows at a time; a B tile of all
    of K (BK == KP) is read once for the group, as v6e's grouped matmul reads each group's weight once. A
    program past the last such group has the table's empty entry and reads nothing."""
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.load(group_ptr + pid)
    nr = tl.load(rows_ptr + g)
    s0 = tl.load(start_ptr + g)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    ncols = rn < N
    rk = tl.arange(0, BK)
    if BK == KP:
        if nr > 0:
            b = tl.load(b_ptr + g * sbg + rk[:, None] * sbk + rn[None, :] * sbn,
                        mask=(rk < K)[:, None] & ncols[None, :], other=0.0)
            for m0 in range(0, nr, BM):
                rows = s0 + m0 + rm
                live = (m0 + rm) < nr
                a = tl.load(a_ptr + rows[:, None] * K + rk[None, :], mask=live[:, None] & (rk < K)[None, :],
                            other=0.0)
                acc = tl.dot(a, b)
                tl.store(c_ptr + rows[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty),
                         mask=live[:, None] & ncols[None, :])
    else:
        for m0 in range(0, nr, BM):
            rows = s0 + m0 + rm
            live = (m0 + rm) < nr
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, KP, BK):
                kk = k0 + rk
                a = tl.load(a_ptr + rows[:, None] * K + kk[None, :], mask=live[:, None] & (kk < K)[None, :],
                            other=0.0)
                b = tl.load(b_ptr + g * sbg + kk[:, None] * sbk + rn[None, :] * sbn,
                            mask=(kk < K)[:, None] & ncols[None, :], other=0.0)
                acc += tl.dot(a, b)
            tl.store(c_ptr + rows[:, None] * N + rn[None, :], acc.to(c_ptr.dtype.element_ty),
                     mask=live[:, None] & ncols[None, :])
