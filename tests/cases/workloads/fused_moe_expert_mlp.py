"""A fused MoE expert MLP -- gather the expert's weights, SwiGLU, weighted combine.

torch equivalent, per token, for its top-1 expert e and routing weight g:
    h   = silu(x @ w_gate[e]) * (x @ w_up[e])
    out = g * (h @ w_down[e])

WHY THIS IS ONE KERNEL AND NOT FIVE
  A MoE layer is the largest weight tensor in the model and each token touches
  only one expert's slice of it. Doing it as separate ops means materialising
  the gathered weights, then three matmuls with two temporaries the size of the
  intermediate -- for arithmetic that is a single pass. Every MoE runtime fuses
  the gather into the matmul and the SwiGLU into the epilogue; this is that
  kernel at one token.

  dl/deepseek_moe.py and dl/moe_router.py cover CHOOSING the expert. This one
  covers what happens after: the routing decision is an input here, so the
  kernel under test is the gather-and-compute, not the argmax.

THE THREE HARD PARTS, AND THEY COMPOSE
  1. THE WEIGHT TILE'S BASE IS DATA. `w_gate[e]` means a [K, N] tile whose
     address came out of memory -- the same indirection as paged_attention.py,
     but on the operand of a DOT rather than of a load. Whether that survives to
     the systolic array is the question.
  2. TWO MATMULS MEET ELEMENTWISE. gate and up are separate dots whose results
     multiply, so both have to be live at once -- dl/swiglu.py in the expert
     case.
  3. A THIRD DOT CONSUMES THE RESULT, so the intermediate feeds straight back
     into the array without going to memory, and the routing weight scales the
     output at the very end.

TOP-1 rather than top-2 so the kernel is the gather and not the accumulation.
Two experts is the same tile twice and a sum, which adds no new shape.

Expert 2 of 4, chosen in the data. A backend that folded the expert index at compile time would read expert 0 and be wrong everywhere.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

E, K, N = 4, 16, 16                     # experts, hidden, intermediate


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fused_moe_expert_mlp_kernel(x_ptr, wg_ptr, wu_ptr, wd_ptr, e_ptr,
                                    g_ptr, o_ptr, K: tl.constexpr,
                                    N: tl.constexpr):
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)

        e = tl.load(e_ptr)                       # which expert -- from memory
        g = tl.load(g_ptr)                       # its routing weight

        # [K, N] tiles whose BASE is data, three of them.
        kn = rk[:, None] * N + rn[None, :]
        nk = rn[:, None] * K + rk[None, :]
        wg = tl.load(wg_ptr + e * (K * N) + kn)
        wu = tl.load(wu_ptr + e * (K * N) + kn)
        wd = tl.load(wd_ptr + e * (N * K) + nk)

        x = tl.load(x_ptr + rk)[None, :]         # [1, K]
        gate = tl.sum(x.T * wg, axis=0)          # [N]
        up = tl.sum(x.T * wu, axis=0)            # [N]
        h = (gate * tl.sigmoid(gate)) * up       # SwiGLU
        tl.store(o_ptr + rk, g * tl.sum(h[:, None] * wd, axis=0))

    return fused_moe_expert_mlp_kernel


def make_inputs(torch, seed=0):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32)
    return {"x": r(K), "wg": r(E, K, N), "wu": r(E, K, N), "wd": r(E, N, K),
            "e": torch.tensor([2], dtype=torch.int32),
            "g": torch.tensor([0.7], dtype=torch.float32)}


def reference(inputs):
    # torch is imported HERE, not at module scope: the pytorchsim-triton-compiler triton venv has no
    # torch and loads this file to build the kernel (pytorchsim_triton_compiler/contract/kernel_spec.py).
    import torch
    e = int(inputs["e"][0])
    x = inputs["x"]
    gate = x @ inputs["wg"][e]
    up = x @ inputs["wu"][e]
    h = torch.nn.functional.silu(gate) * up
    return {"o": inputs["g"][0] * (h @ inputs["wd"][e])}


SPEC = KernelSpec(
    name="fused_moe_expert_mlp_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "wg_ptr": "*fp32", "wu_ptr": "*fp32",
               "wd_ptr": "*fp32", "e_ptr": "*i32", "g_ptr": "*fp32",
               "o_ptr": "*fp32", "K": "constexpr", "N": "constexpr"},
    constexprs={"K": K, "N": N},
    args=[Arg("x", IN, "float32", (K,)),
          Arg("wg", IN, "float32", (E, K, N)),
          Arg("wu", IN, "float32", (E, K, N)),
          Arg("wd", IN, "float32", (E, N, K)),
          Arg("e", IN, "int32", (1,)), Arg("g", IN, "float32", (1,)),
          Arg("o", OUT, "float32", (K,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
