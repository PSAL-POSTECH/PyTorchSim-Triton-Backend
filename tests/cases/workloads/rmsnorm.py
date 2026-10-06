"""RMSNorm -- the normaliser every current open LLM actually uses.

    y = x * rsqrt(mean(x*x) + eps) * w

Llama, Mistral, Qwen, Gemma and Phi all use this and not LayerNorm, because it
drops the mean subtraction and the bias for no measured quality cost. So it is
the norm to be correct on, and it is CHEAPER TO GET WRONG QUIETLY than
LayerNorm: with no `d = h - mean` in front of it, a scale that is off by a
constant factor still produces plausible-looking activations.

WHAT IT EXERCISES HERE
  One reduction whose result is read back over the row -- `tl.sum(x*x, axis=1)`
  then `x * rsqrt(...)[:, None]`. On a rank-2 tile with the fold on the INNER
  axis that costs nothing: each lane owns a whole row, so the fold and the splat
  both stay inside one bank. That is the arrangement to reach for, and this
  kernel is what says so is still true.

  `tl.sum(x * x)` also has the multiply FUSED INTO the reduction body by
  fuse, so the fold's body computes before it combines.

  The second arm folds the same data the other way (`axis=0`, a column norm) so
  the two lane-axis answers for one buffer meet in one kernel. Nothing in an LLM
  normalises down a column; it is here because the layout question is the same
  one and only one of the two is free.

torch equivalent: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

The norm every current open LLM uses. One reduction read back over the row, and the same data folded the other way so both lane-axis answers for one buffer meet in one kernel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 64
EPS = 1e-5


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rmsnorm_kernel(x_ptr, w_ptr, o_ptr, c_ptr,
                       N: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr):
        rows = tl.arange(0, BM)
        cols = tl.arange(0, N)
        idx = rows[:, None] * N + cols[None, :]
        x = tl.load(x_ptr + idx)

        ms = tl.sum(x * x, axis=1) / N                 # per row, on the lanes
        y = x * tl.rsqrt(ms[:, None] + EPS)
        w = tl.load(w_ptr + cols)
        tl.store(o_ptr + idx, y * w[None, :])

        cs = tl.sum(x * x, axis=0) / BM                # per column, across them
        tl.store(c_ptr + cols, tl.rsqrt(cs + EPS))

    return rmsnorm_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32),
            "w": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(M, N)
    w = inputs["w"]
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
    c = torch.rsqrt(x.pow(2).mean(0) + EPS)
    return {"o": y.reshape(-1), "c": c}


SPEC = KernelSpec(
    name="rmsnorm_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "c_ptr": "*fp32", "N": "constexpr", "BM": "constexpr",
               "EPS": "constexpr"},
    constexprs={"N": N, "BM": M, "EPS": EPS},
    args=[Arg("x", IN, "float32", (M * N,)), Arg("w", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (M * N,)), Arg("c", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
