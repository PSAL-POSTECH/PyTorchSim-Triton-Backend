"""Flash attention in the SDPA TEMPLATE's algebraic form -- the same maths, the
other way round, to measure what the form alone is worth.

torch equivalent: softmax(q @ k.T) @ v, identical to workloads/attention.py.

WHY IT EXISTS.  attention.py spends 5 DRAM round trips and the round log says all
of them are forced: one axis group holds two matrix-unit MUSTs (`no axis every
matmul can stream`), because S is the SECOND matmul's first operand and the
FIRST matmul's result, and the array wants a different axis of it for each.
PyTorchSim's own `mlir_sdpa_template.py` does not hit that -- it writes
`mul = key @ query.t` and then `out.t = value.t @ mul`, so both matmuls read the
shared tile the same way round, and it says so in as many words: "the first
operand (key) must be split along the column axis" and "output is also stored in
transposed format to match the value.t @ Softmax(mul) operation".

This kernel is that form, written in Triton:

    St  = k @ q.t          (BN, BM)   scores transposed
    p   = softmax(St, axis=0)         over KEYS, which is St's ROW axis
    acc.t = v.t @ p        (D,  BM)   output transposed
    store acc.t.T

If the round trips are a property of the FORM and not of the compiler, this
compiles with fewer of them than attention.py on the same tree. If it does not,
the algebraic rewrite is not the missing free move and the search moves on.

attention.py's maths in the SDPA template's transposed form, to measure what the algebraic form alone is worth in round trips.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M_CTX = 256
N_CTX = 256
D = 128
BM = 128
BN = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def attention_transposed_form_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                                         N_CTX: tl.constexpr, D: tl.constexpr,
                                         BM: tl.constexpr, BN: tl.constexpr,
                                         SCALE: tl.constexpr):
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, BN)

        # (D, BM): q read transposed by its own addressing.
        qt = tl.load(q_ptr + offs_m[None, :] * D + offs_d[:, None])

        m_i = tl.full((BM,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)
        acct = tl.zeros((D, BM), dtype=tl.float32)

        for start_n in range(0, N_CTX, BN):
            n = start_n + offs_n
            k = tl.load(k_ptr + n[:, None] * D + offs_d[None, :])
            # (D, BN): v read transposed, the template's value.t
            vt = tl.load(v_ptr + n[None, :] * D + offs_d[:, None])

            qkt = tl.dot(k, qt) * SCALE          # (BN, BM), scores transposed
            m_new = tl.maximum(m_i, tl.max(qkt, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qkt - m_new[None, :])

            l_i = l_i * alpha + tl.sum(p, axis=0)
            acct = acct * alpha[None, :] + tl.dot(vt, p)
            m_i = m_new

        tl.store(o_ptr + offs_m[None, :] * D + offs_d[:, None],
                 acct / l_i[None, :])

    return attention_transposed_form_kernel


SCALE = 1.0 / (D ** 0.5)


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn(M_CTX * D, generator=g, dtype=torch.float32) * 0.5,
        "k": torch.randn(N_CTX * D, generator=g, dtype=torch.float32) * 0.5,
        "v": torch.randn(N_CTX * D, generator=g, dtype=torch.float32) * 0.5,
    }


def reference(inputs):
    import torch
    q = inputs["q"].reshape(M_CTX, D)
    k = inputs["k"].reshape(N_CTX, D)
    v = inputs["v"].reshape(N_CTX, D)
    s = (q @ k.t()) * SCALE
    p = torch.softmax(s, dim=1)
    return {"o": (p @ v).reshape(-1)}


SPEC = KernelSpec(
    name="attention_transposed_form_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "o_ptr": "*fp32", "N_CTX": "constexpr", "D": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "SCALE": "constexpr"},
    constexprs={"N_CTX": N_CTX, "D": D, "BM": BM, "BN": BN, "SCALE": SCALE},
    args=[Arg("q", IN, "float32", (M_CTX * D,)),
          Arg("k", IN, "float32", (N_CTX * D,)),
          Arg("v", IN, "float32", (N_CTX * D,)),
          Arg("o", OUT, "float32", (M_CTX * D,))],
    grid=(M_CTX // BM,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
