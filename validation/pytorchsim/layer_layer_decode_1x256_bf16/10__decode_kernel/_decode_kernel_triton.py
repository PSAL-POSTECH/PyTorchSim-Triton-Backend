import triton
import triton.language as tl
@triton.jit
def _decode_kernel(q_ptr, k_ptr, v_ptr, o_ptr, HKV: tl.constexpr, G: tl.constexpr,
                   GP: tl.constexpr, S: tl.constexpr, D: tl.constexpr, BKV: tl.constexpr,
                   SCALE: tl.constexpr):
    pid = tl.program_id(0)
    rg = tl.arange(0, GP)
    rd = tl.arange(0, D)
    rows = rg[:, None] < G
    qbase = pid * (G * D)
    kbase = pid * (S * D)
    q = tl.load(q_ptr + qbase + rg[:, None] * D + rd[None, :], mask=rows, other=0.0)
    m_i = tl.full((GP,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((GP,), dtype=tl.float32)
    acc = tl.zeros((GP, D), dtype=tl.float32)
    for start in range(0, S, BKV):
        t = start + tl.arange(0, BKV)
        kt = tl.load(k_ptr + kbase + t[None, :] * D + rd[:, None])
        v = tl.load(v_ptr + kbase + t[:, None] * D + rd[None, :])
        qk = tl.dot(q, kt, out_dtype=tl.float32) * SCALE
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new
    tl.store(o_ptr + qbase + rg[:, None] * D + rd[None, :],
             (acc / l_i[:, None]).to(o_ptr.dtype.element_ty), mask=rows)
