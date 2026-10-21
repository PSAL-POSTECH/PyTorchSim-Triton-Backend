import triton
import triton.language as tl
@triton.jit
def _prefill_kernel(q_ptr, k_ptr, v_ptr, o_ptr, HQ: tl.constexpr, G: tl.constexpr,
                    S: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr, BKV: tl.constexpr,
                    CAUSAL: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // HQ
    h = pid % HQ
    qbase = pid * (S * D)
    kbase = (b * (HQ // G) + h // G) * (S * D)
    rn = tl.arange(0, BKV)
    rd = tl.arange(0, D)
    for qb in tl.static_range(S // BQ):
        rm = qb * BQ + tl.arange(0, BQ)
        q = tl.load(q_ptr + qbase + rm[:, None] * D + rd[None, :])
        m_i = tl.full((BQ,), -1.0e30, dtype=tl.float32)
        l_i = tl.zeros((BQ,), dtype=tl.float32)
        acc = tl.zeros((BQ, D), dtype=tl.float32)
        for start in range(0, qb * BQ if CAUSAL else S, BKV):
            n = start + rn
            kt = tl.load(k_ptr + kbase + n[None, :] * D + rd[:, None])
            v = tl.load(v_ptr + kbase + n[:, None] * D + rd[None, :])
            qk = tl.dot(q, kt, out_dtype=tl.float32) * SCALE
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new
        if CAUSAL:
            n = qb * BQ + rn
            kt = tl.load(k_ptr + kbase + n[None, :] * D + rd[:, None])
            v = tl.load(v_ptr + kbase + n[:, None] * D + rd[None, :])
            qk = tl.dot(q, kt, out_dtype=tl.float32) * SCALE
            qk = tl.where(rm[:, None] >= n[None, :], qk, -1.0e30)
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new
        tl.store(o_ptr + qbase + rm[:, None] * D + rd[None, :],
                 (acc / l_i[:, None]).to(o_ptr.dtype.element_ty))
