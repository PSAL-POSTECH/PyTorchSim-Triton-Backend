"""GPTQ with `desc_act=True` -- the activations are PERMUTED before the matmul."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N, G = 16, 32, 16, 8            # tokens, hidden, out channels, group size


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gptq_actorder_gather_kernel(x_ptr, perm_ptr, wq_ptr, s_ptr,
                                    xp_ptr, o_ptr, M: tl.constexpr,
                                    K: tl.constexpr, N: tl.constexpr,
                                    G: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)

        # THE ACT-ORDER GATHER. `p` is a runtime permutation of [0, K), so the
        # inner offset of this 2-D load is a loaded tensor and not an affine
        # function of the iteration index. The row offset stays affine.
        p = tl.load(perm_ptr + rk)                                   # [K] int32
        xp = tl.load(x_ptr + rm[:, None] * K + p[None, :])           # [M, K]
        tl.store(xp_ptr + rm[:, None] * K + rk[None, :], xp)

        # From here down this is quant_int4_matmul.py verbatim -- and it only
        # works because the permutation already made the groups contiguous.
        byte = tl.load(wq_ptr + (rk[:, None] // 2) * N + rn[None, :])
        w = ((byte >> ((rk[:, None] % 2) * 4)) & 0xF) - 8             # [K, N]
        s = tl.load(s_ptr + (rk[:, None] // G) * N + rn[None, :])
        wf = w.to(tl.float32) * s

        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(xp, wf))

    return gptq_actorder_gather_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, K, generator=g, dtype=torch.float32),
            # A genuine permutation: the argsort of the per-row activation norm
            # is what GPTQ stores, and any permutation exercises the same load.
            "perm": torch.randperm(K, generator=g).to(torch.int32),
            "wq": torch.randint(0, 256, (K // 2, N), generator=g,
                                dtype=torch.int32),
            "s": torch.rand(K // G, N, generator=g, dtype=torch.float32) * 0.1}


def reference(inputs):
    import torch
    xp = inputs["x"][:, inputs["perm"].to(torch.int64)]
    byte = inputs["wq"]
    lo = (byte & 0xF) - 8
    hi = ((byte >> 4) & 0xF) - 8
    w = torch.stack([lo, hi], dim=1).reshape(K, N).to(torch.float32)
    w = w * inputs["s"].repeat_interleave(G, dim=0)
    return {"xp": xp.contiguous(), "o": xp @ w}


SPEC = KernelSpec(
    name="gptq_actorder_gather_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "perm_ptr": "*i32", "wq_ptr": "*i32",
               "s_ptr": "*fp32", "xp_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr", "N": "constexpr",
               "G": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "G": G},
    args=[Arg("x", IN, "float32", (M, K)),
          Arg("perm", IN, "int32", (K,)),
          Arg("wq", IN, "int32", (K // 2, N)),
          Arg("s", IN, "float32", (K // G, N)),
          Arg("xp", OUT, "float32", (M, K)),
          Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-passes/04-pytorchsim-materialize-layouts.mlir": {"pytorchsim.copy": 0}},
)
