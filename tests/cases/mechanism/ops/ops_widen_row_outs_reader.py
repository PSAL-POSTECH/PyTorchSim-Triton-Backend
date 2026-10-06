"""A row a broadcast wants WIDE and another generic WRITES INTO.

WHAT IT PINS, AND IT IS NO LONGER A PASS'S READER CHECK. `rc * 2.0` comes down
as `ins(%rc) outs(%rc)` -- triton-shared builds a unary elementwise result on its
own input -- so `rc` is not just read by that op, it is what the op's RESULT IS
TYPED AS. widen_broadcast_loads widened the row by rewriting its type, which
therefore widened that result, and then everything reading the result, and the
walk did not stop:

    'linalg.generic' op expected type of operand #2 ('tensor<16x16xf32>') to
    match type of corresponding result ('tensor<1x16xf32>')

THE PASS IS GONE and this is the other half of why that was safe. Its
replacement, fit_layout_to_lanes._clone_loaded_row_wide, retypes nothing: a second
buffer is allocated, a second transfer reads the same DRAM into it with a zero
stride, and ONE operand is re-pointed. An `outs` slot that happens to hold the
row is untouched and keeps its own type, so the recursion this file was written
about cannot start. It stays as the shape that would start it.

WHY IT NEEDS ITS OWN FILE beside ops_widen_row_narrow_reader: same defect, and
linalg names it with a different verifier message depending on which half it
notices first -- an operand/result mismatch here, an inferred iteration extent
there. A staging that retypes would trip exactly one of them.

torch equivalent: (x + c, c * 2).

A reader that builds its result ON the row is not a reader that can be told the row grew: the result grows with it and the group does not end there.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 16, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_widen_row_outs_reader_kernel(x_ptr, c_ptr, o_ptr, m_ptr,
                                         M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rc = tl.load(c_ptr + rn[None, :])                        # (1, N)
        nc = rc * 2.0             # `rc` in the OUTS slot of a narrow generic
        tile = tl.load(x_ptr + rm[:, None] * N + rn[None, :])    # (M, N)
        out = tile + rc           # ... and the source of a broadcast
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], out)
        tl.store(m_ptr + rn[None, :], nc)

    return ops_widen_row_outs_reader_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32),
            "c": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, N)
    c = inputs["c"].reshape(1, N)
    return {"o": (x + c).reshape(-1), "m": (c * 2.0).reshape(-1)}


SPEC = KernelSpec(
    name="ops_widen_row_outs_reader_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "c_ptr": "*fp32", "o_ptr": "*fp32",
               "m_ptr": "*fp32", "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)), Arg("c", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (M * N,)), Arg("m", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
