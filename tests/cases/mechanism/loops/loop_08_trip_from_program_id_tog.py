"""Pins that a loop whose trip count is the program id, holding DMAs, is blocked at the TOG: a TOG holds counted
loops only. This is causal flash attention gridded over query blocks (the KV loop of block i runs i blocks), the form
splash and the Triton tutorial take; validation/attention moves the query blocks into the program as a static loop."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

S, D, B = 256, 128, 128
SCALE = 1.0 / D ** 0.5


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_08_trip_from_program_id_tog(q_ptr, k_ptr, v_ptr, o_ptr, D: tl.constexpr, B: tl.constexpr,
                                         SCALE: tl.constexpr):
        pm = tl.program_id(0)
        rm = pm * B + tl.arange(0, B)
        rn = tl.arange(0, B)
        rd = tl.arange(0, D)
        q = tl.load(q_ptr + rm[:, None] * D + rd[None, :])
        m_i = tl.full((B,), -1.0e30, dtype=tl.float32)
        l_i = tl.zeros((B,), dtype=tl.float32)
        acc = tl.zeros((B, D), dtype=tl.float32)
        for start in range(0, (pm + 1) * B, B):
            n = start + rn
            kt = tl.load(k_ptr + n[None, :] * D + rd[:, None])
            v = tl.load(v_ptr + n[:, None] * D + rd[None, :])
            qk = tl.dot(q, kt, out_dtype=tl.float32) * SCALE
            qk = tl.where(rm[:, None] >= n[None, :], qk, -1.0e30)
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
            m_i = m_new
        tl.store(o_ptr + rm[:, None] * D + rd[None, :], (acc / l_i[:, None]).to(tl.bfloat16))

    return loop_08_trip_from_program_id_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {x: torch.randn(S, D, generator=g).to(torch.bfloat16) for x in "qkv"}


def reference(inputs):
    import torch
    q, k, v = (inputs[x].float() for x in "qkv")
    s = (q @ k.T * SCALE).masked_fill(torch.ones(S, S, dtype=torch.bool).triu(1), float("-inf"))
    return {"o": (torch.softmax(s, -1) @ v).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="loop_08_trip_from_program_id_tog", kernel=kernel,
    signature={"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "o_ptr": "*bf16",
               "D": "constexpr", "B": "constexpr", "SCALE": "constexpr"},
    constexprs={"D": D, "B": B, "SCALE": SCALE},
    args=[Arg(x, IN, "bfloat16", (S, D)) for x in "qkv"] + [Arg("o", OUT, "bfloat16", (S, D))],
    grid=(S // B,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    expect="blocked",
    expect_stop=Stop("tog", Reason.TOG_UNCOUNTED_LOOP), expect_stage="togcheck",
    expect_why="the KV loop runs program_id + 1 blocks and holds DMAs; the TOG builder replays counted "
               "loops only (build_tog refuse_unreplayed)",
)
