"""ONE row read as dim 0 AND as dim 1 by one op -- two axes, not one.

    d[i, j] = r[i] - r[j]          one memref<Nxi32>, two indexing maps
    out     = (d * col[None, :]) / sum(d, axis=1)[:, None]

WHAT IT PINS. `lib_axis_graph` reads a value's participation per USE. The
generic above has `ins(%r, %r)` with `(d0,d1)->(d0)` and `(d0,d1)->(d1)`, so
axis 0 of ONE buffer names iteration dim 0 at one slot and iteration dim 1 at
the other. Reading that per VALUE joins node (%r, 0) to dim 0's class and to
dim 1's class, which UNIONS THEM -- the graph then asserts that the two
iteration axes of a rank-2 op are one axis, which is false of every op that has
two.

The whole function collapses into a single axis class when it does. That class
is refused, and no arrangement of the values that exist can answer it, so every
DRAM round trip retires nothing and the demand comes back on the copy.

    measured   on the tree before this, with `_add_op` reading per value:
               `NotVectorisable: one operand of this elementwise op is in a
               single bank (ONE_LANE) while another is banked across`
               dl/rmsnorm and dl/fused_add_rmsnorm fail with the same sentence,
               and dl/alibi ships 16 round trips, 8 of them POINTLESS, against
               8 and 0 after.

WHAT BREAKS IT AGAIN. Any change that unions the two reads back into one class
-- keying a graph node on (value, axis) alone, or "canonicalising" the second
use onto the first. The row genuinely has one lane axis and the two reads
genuinely want different ones; that is a rule 17 CONTRADICTION, and it is
answered by making a second value (the row staged wide), never by agreeing.

THE COLUMN BROADCAST IS WHY THIS IS NOT dl/alibi. alibi's difference feeds only
reductions along one axis, so the merged class still has an answer it can limp
to. Here a second row is broadcast along the OTHER axis, so the merged class has
a NOT on every axis it has and the function has to pick between two live
demands.

torch equivalent:
    r = torch.arange(N)
    d = (r[:, None] - r[None, :]).float()
    out = d * col[None, :] / d.sum(1)[:, None]

one buffer axis read as two iteration dims by one linalg.generic: a rule 17 contradiction, not one axis class.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def row_two_dims_kernel(c_ptr, w_ptr, o_ptr, M: tl.constexpr,
                            N: tl.constexpr):
        i = tl.arange(0, M)
        j = tl.arange(0, N)
        # ONE rank-1 value, read down BOTH iteration dims of one rank-2 op.
        d = (i[:, None] - j[None, :]).to(tl.float32)
        x = tl.load(c_ptr + i[:, None] * N + j[None, :])
        b = x * d + 1.0
        # ONE fold per axis, so the merged class has a NOT on everything it has.
        r = tl.sum(b, axis=1)
        c = tl.sum(b, axis=0)
        # A LOADED ROW along one of the two, so the class the merged graph
        # produced has a pin on it as well as a fold.
        w = tl.load(w_ptr + j)
        tl.store(o_ptr + i[:, None] * N + j[None, :],
                 b * w[None, :] / (r[:, None] + c[None, :]))

    return row_two_dims_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"c": torch.randn(M * N, generator=g, dtype=torch.float32),
            "w": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    d = (torch.arange(M)[:, None] - torch.arange(N)[None, :]).float()
    b = inputs["c"].reshape(M, N) * d + 1.0
    out = b * inputs["w"][None, :] / (b.sum(1)[:, None] + b.sum(0)[None, :])
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="row_two_dims_kernel",
    kernel=kernel,
    signature={"c_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("c", IN, "float32", (M * N,)),
          Arg("w", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
