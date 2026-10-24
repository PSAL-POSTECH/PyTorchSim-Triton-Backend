"""DeepSeek-V3's MoE -- shared experts, sigmoid routing, and a balancing bias."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, E, D = 64, 8, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def deepseek_moe_kernel(x_ptr, wg_ptr, b_ptr, sh_ptr, w_ptr, o_ptr,
                            T: tl.constexpr, E: tl.constexpr,
                            D: tl.constexpr):
        t = tl.arange(0, T)
        e = tl.arange(0, E)
        d = tl.arange(0, D)
        x = tl.load(x_ptr + t[:, None] * D + d[None, :])

        # AFFINITIES, one expert at a time so every tile stays rank 2.
        s = tl.zeros((T, E), dtype=tl.float32)
        for jx in tl.static_range(E):
            wj = tl.load(wg_ptr + jx * D + d)
            lj = tl.sum(x * wj[None, :], axis=1)
            s = tl.where(e[None, :] == jx, lj[:, None], s)
        s = tl.sigmoid(s)              # per expert, not against the others

        # RANK BY s + bias, WEIGHT BY s. The bias only ever enters the choice.
        r = s + tl.load(b_ptr + e)[None, :]
        r1 = tl.max(r, axis=1)
        i1 = tl.argmax(r, axis=1)
        r_2 = tl.where(e[None, :] == i1[:, None], -1e30, r)
        i2 = tl.argmax(r_2, axis=1)

        # The weights come from `s` at the positions `r` chose -- two rank-1
        # vectors kept aligned through one selection.
        g1 = tl.sum(tl.where(e[None, :] == i1[:, None], s, 0.0), axis=1)
        g2 = tl.sum(tl.where(e[None, :] == i2[:, None], s, 0.0), axis=1)
        tot = g1 + g2
        g1 = g1 / tot
        g2 = g2 / tot

        # SHARED first, then the two routed ones. The shared expert needs no
        # index at all -- every token uses it.
        out = x * tl.load(sh_ptr + d)[None, :]
        out += g1[:, None] * tl.load(w_ptr + i1[:, None] * D + d[None, :]) * x
        out += g2[:, None] * tl.load(w_ptr + i2[:, None] * D + d[None, :]) * x
        tl.store(o_ptr + t[:, None] * D + d[None, :], out + r1[:, None] * 0.0)

    return deepseek_moe_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32),
            "wg": torch.randn(E * D, generator=g, dtype=torch.float32),
            "b": torch.randn(E, generator=g, dtype=torch.float32) * 0.2,
            "sh": torch.randn(D, generator=g, dtype=torch.float32),
            "w": torch.randn(E * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    s = torch.sigmoid(x @ inputs["wg"].reshape(E, D).T)
    r = s + inputs["b"][None, :]
    idx = r.topk(2, dim=1).indices
    gv = s.gather(1, idx)
    gv = gv / gv.sum(1, keepdim=True)
    w = inputs["w"].reshape(E, D)
    out = x * inputs["sh"][None, :]
    out = out + gv[:, 0:1] * w[idx[:, 0]] * x + gv[:, 1:2] * w[idx[:, 1]] * x
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="deepseek_moe_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "wg_ptr": "*fp32", "b_ptr": "*fp32",
               "sh_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "T": "constexpr", "E": "constexpr", "D": "constexpr"},
    constexprs={"T": T, "E": E, "D": D},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("wg", IN, "float32", (E * D,)),
          Arg("b", IN, "float32", (E,)),
          Arg("sh", IN, "float32", (D,)),
          Arg("w", IN, "float32", (E * D,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-passes/04-pytorchsim-materialize-layouts.mlir": {"pytorchsim.copy": 0}},
)
