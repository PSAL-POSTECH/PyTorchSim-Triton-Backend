"""A reduction over an axis of extent 1, which crashes vectorize.

    tl.sum(v * v, axis=1)   over a [16, 1, 512] tile

FOUND while trying to build a kernel for `absorb_layout_ops`'s unit-axis move; it is
a different defect and gets its own file (rule 2). Stage 4:

    File "pytorchsim_triton_compiler/elf/passes/p05_vectorize.py", line 267, in _reduced_dim
        return real[0] if real else dims[0]
    IndexError: list index out of range

`_reduced_dim` picks the axis a reduce folds and falls back to `dims[0]` when no
axis is "real". Here BOTH lists are empty: the reduction axis has extent 1, so
`--linalg-fold-unit-extent-dims` upstream has already folded it away, and what
reaches vectorize is a reduce with nothing left to reduce over. The fallback is
written for "no real axis", not for "no axis", and the difference is a crash
rather than a diagnosis.

GREEN SINCE 2026-09-05, err 0, AND `_reduced_dim` WAS NOT TOUCHED. The guess
above was right about where: absorb_layout_ops is what strikes the unit axis out
of `dimensions`, so absorb_layout_ops is what owes the retirement. A reduce with
an empty `dimensions` is not a reduction at all -- `linalg.reduce` then says
`out = combine(in, init)` per element, which is a generic -- and it is rewritten
as one there. Making `dims[0]` safe in vectorize would have been the silent
wrong axis this file warned about.

The kernel also declared `m` and `n` as runtime scalars and gave the harness no
values for them. Nothing had ever reached stage 6 to say so.

NOT A REGRESSION. Measured on the tree before and after
absorb_layout_ops's `_dims_the_move_retires` change: identical crash, so it predates that
work and is not caused by it.

ARITHMETIC. Would be `x * x` under a two-axis mask, exact in f32, if it ran.

a reduce over an extent-1 axis: upstream folds it away and the reduce that arrives folds nothing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 16
N = 512
MBOUND = 11
NBOUND = 300


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_reduce_over_unit_axis_kernel(x_ptr, o_ptr, m, n,
                                          M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)[:, None, None]
        rk = tl.arange(0, 1)[None, :, None]
        rn = tl.arange(0, N)[None, None, :]
        v = tl.load(x_ptr + rm * N + rk * N + rn,
                    mask=(rm < m) & (rn < n), other=0.0)
        # AXIS 1 HAS EXTENT 1 -- the reduction is the identity, and that is the
        # whole kernel.
        w = tl.sum(v * v, axis=1)
        tl.store(o_ptr + tl.arange(0, M)[:, None] * N
                 + tl.arange(0, N)[None, :], w)

    return tile_reduce_over_unit_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(M, N)
    keep = ((torch.arange(M).reshape(M, 1) < MBOUND)
            & (torch.arange(N).reshape(1, N) < NBOUND))
    v = torch.where(keep, x, torch.zeros_like(x))
    return {"o": v * v}


SPEC = KernelSpec(
    name="tile_reduce_over_unit_axis_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "m": "i32", "n": "i32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)),
          Arg("o", OUT, "float32", (M, N))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # `m` and `n` are the mask's bounds and they are RUNTIME scalars, so the
    # harness has to be given values for them: the signature alone makes the
    # MLIR function take an argument that nothing passes.
    extra={"scalar_args": [("m", "int32_t"), ("n", "int32_t")],
           "scalar_values": {"m": MBOUND, "n": NBOUND}},
)
