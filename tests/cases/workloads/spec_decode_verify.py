"""Speculative decoding's acceptance test -- and the residual that makes it exact."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, V = 16, 64                   # drafted positions, vocabulary
FLOOR = 1e-9                    # guards the residual's normaliser


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def spec_decode_verify_kernel(p_ptr, q_ptr, tok_ptr, u_ptr,
                                  a_ptr, n_ptr, r_ptr,
                                  K: tl.constexpr, V: tl.constexpr,
                                  FLOOR: tl.constexpr):
        # LAYOUT ONE: positions in-lane, because the scan runs over positions.
        j = tl.arange(0, K)[None, :]
        tok = tl.load(tok_ptr + j)
        pt = tl.load(p_ptr + j * V + tok)          # target's mass on the draft
        qt = tl.load(q_ptr + j * V + tok)          # draft's own mass on it

        # Cross-multiplied, so a vanishing draft probability cannot divide.
        ok = tl.load(u_ptr + j) * qt <= pt
        # THE PREFIX: a position survives only if nothing before it failed.
        pre = tl.cumsum(tl.where(ok, 0.0, 1.0), axis=1) == 0.0
        tl.store(a_ptr + j, pre.to(tl.int32))
        n = tl.sum(tl.where(pre, 1.0, 0.0), axis=1)              # (1,)
        tl.store(n_ptr + tl.arange(0, 1), n.to(tl.int32))

        # LAYOUT TWO: positions on the lanes, because the reduction is over V.
        rk = tl.arange(0, K)[:, None]
        rv = tl.arange(0, V)[None, :]
        off = rk * V + rv
        d = tl.maximum(tl.load(p_ptr + off) - tl.load(q_ptr + off), 0.0)
        d = d / tl.maximum(tl.sum(d, axis=1), FLOOR)[:, None]

        # `n` selects a row by COMPARISON, never by address -- it has to stay in
        # a vector. Nothing matches when every position was accepted.
        tl.store(r_ptr + off, tl.where(rk == n[None, :], d, 0.0))

    return spec_decode_verify_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Real distributions on both sides, and a draft that is close but not equal
    # to the target -- which is the regime the acceptance rate is decided in.
    base = torch.randn(K, V, generator=g, dtype=torch.float32) * 2.0
    p = torch.softmax(base, dim=1)
    q = torch.softmax(base + torch.randn(K, V, generator=g,
                                         dtype=torch.float32) * 0.8, dim=1)
    return {"p": p.reshape(-1).contiguous(),
            "q": q.reshape(-1).contiguous(),
            # what the draft sampled: from q, so most positions are acceptable
            "tok": torch.multinomial(q, 1, generator=g).reshape(-1).to(
                torch.int32),
            "u": torch.rand(K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    p = inputs["p"].reshape(K, V)
    q = inputs["q"].reshape(K, V)
    j = torch.arange(K)
    t = inputs["tok"].long()
    ok = inputs["u"] * q[j, t] <= p[j, t]
    pre = (~ok).to(torch.int32).cumsum(0) == 0
    n = int(pre.sum())
    d = (p - q).clamp_min(0.0)
    d = d / d.sum(dim=1, keepdim=True).clamp_min(FLOOR)
    sel = (j == n)[:, None]
    return {"a": pre.to(torch.int32).contiguous(),
            "n": torch.tensor([n], dtype=torch.int32),
            "r": torch.where(sel, d, torch.zeros_like(d)).reshape(
                -1).contiguous()}


SPEC = KernelSpec(
    name="spec_decode_verify_kernel", kernel=kernel,
    signature={"p_ptr": "*fp32", "q_ptr": "*fp32", "tok_ptr": "*i32",
               "u_ptr": "*fp32", "a_ptr": "*i32", "n_ptr": "*i32",
               "r_ptr": "*fp32", "K": "constexpr", "V": "constexpr",
               "FLOOR": "constexpr"},
    constexprs={"K": K, "V": V, "FLOOR": FLOOR},
    args=[Arg("p", IN, "float32", (K * V,)),
          Arg("q", IN, "float32", (K * V,)),
          Arg("tok", IN, "int32", (K,)),
          Arg("u", IN, "float32", (K,)),
          Arg("a", OUT, "int32", (K,)),
          Arg("n", OUT, "int32", (1,)),
          Arg("r", OUT, "float32", (K * V,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
    expect_ir={"08-passes/04-pytorchsim-materialize-layouts.mlir": {"torchsim.copy": 9}},
)
