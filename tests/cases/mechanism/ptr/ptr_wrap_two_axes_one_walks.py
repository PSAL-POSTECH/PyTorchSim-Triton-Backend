"""Two axes wrap, the third still walks.

    idx = (r % 3)*S*T + (s % 3)*T + t        axes 0 and 1 wrap, axis 2 plain

WHY THIS SHAPE. `shape` has a slot per axis, so a wrap costs THAT axis's dram
stride and nothing else -- `dram_stride = [0, 0, 1]` here. A modulus on the
flattened index would cost all three at once, which is the difference between
holding the wrap per axis and holding it on the sum.

Two axes is also the case addState refused until 2026-08-31: one `hasModulo()`
above a loop that was already per-axis. The mm store in
inductor_mm_modulo.py is the shape that made it matter; this one is where the
saving is visible, because rank 2 has no third axis left to walk.

R=8 over 3 and S=4 over 3, so both wraps fire.

Two wrapped axes and one plain one: the plain axis keeps its dram stride, which is what a per-axis wrap buys over a flat one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, S, T, MR, MS = 8, 4, 8, 3, 3
ASIZE = 256


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_two_axes_one_walks(a_ptr, o_ptr, R: tl.constexpr,
                                S: tl.constexpr, T: tl.constexpr,
                                MR: tl.constexpr, MS: tl.constexpr):
        r = tl.arange(0, R) % MR
        s = tl.arange(0, S) % MS
        t = tl.arange(0, T)
        v = tl.load(a_ptr + (r[:, None, None] * (S * T)
                             + s[None, :, None] * T + t[None, None, :]))
        tl.store(o_ptr + (tl.arange(0, R)[:, None, None] * (S * T)
                          + tl.arange(0, S)[None, :, None] * T
                          + t[None, None, :]), v)

    return wrap_two_axes_one_walks


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(ASIZE, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"]
    r = torch.arange(R) % MR
    s = torch.arange(S) % MS
    t = torch.arange(T)
    idx = (r[:, None, None] * (S * T) + s[None, :, None] * T
           + t[None, None, :])
    return {"o": a[idx].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="wrap_two_axes_one_walks", kernel=kernel,
    signature={"a_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr",
               "S": "constexpr", "T": "constexpr", "MR": "constexpr",
               "MS": "constexpr"},
    constexprs={"R": R, "S": S, "T": T, "MR": MR, "MS": MS},
    args=[Arg("a", IN, "float32", (ASIZE,)),
          Arg("o", OUT, "float32", (R * S * T,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "tts.make_tptr": 2}},
)
