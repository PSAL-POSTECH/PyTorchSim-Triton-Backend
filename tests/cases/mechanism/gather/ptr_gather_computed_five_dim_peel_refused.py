"""A rank-5 gather with a computed index, so no indirect_dim: a peel's pieces would each read the
index buffer from its start; refused. The old tree peeled it, right only because the index is
constant along the peeled axis. Breaks it: peeling an indirect transfer with no indirect_dim."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

ROWS = 256
A, B, C, D = 4, 4, 4, 16
BM, BA, BB, BC, BD = 128, 2, 2, 2, 8
W = A * B * C * D


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_computed_five_dim_peel_refused_kernel(x_ptr, o_ptr, ROWS: tl.constexpr,
                                          B: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
                                          W: tl.constexpr, BM: tl.constexpr, BA: tl.constexpr,
                                          BB: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr):
        rm = tl.arange(0, BM)
        ra = tl.arange(0, BA)
        rb = tl.arange(0, BB)
        rc = tl.arange(0, BC)
        rd = tl.arange(0, BD)
        idx = (rm * 37 + 11) % ROWS
        inner = (ra[None, :, None, None, None] * (B * C * D)
                 + rb[None, None, :, None, None] * (C * D)
                 + rc[None, None, None, :, None] * D
                 + rd[None, None, None, None, :])
        v = tl.load(x_ptr + idx[:, None, None, None, None] * W + inner)
        tl.store(o_ptr + rm[:, None, None, None, None] * W + inner, v)

    return ptr_gather_computed_five_dim_peel_refused_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS * W, generator=g, dtype=torch.float32),
            "o": torch.zeros(BM * W, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = (torch.arange(BM) * 37 + 11) % ROWS
    s = inputs["x"].view(ROWS, A, B, C, D)[i]
    o = inputs["o"].clone().view(BM, A, B, C, D)
    o[:, :BA, :BB, :BC, :BD] = s[:, :BA, :BB, :BC, :BD]
    return {"o": o.reshape(-1)}


SPEC = KernelSpec(
    name="ptr_gather_computed_five_dim_peel_refused_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "ROWS": "constexpr",
               "B": "constexpr", "C": "constexpr", "D": "constexpr", "W": "constexpr",
               "BM": "constexpr", "BA": "constexpr", "BB": "constexpr", "BC": "constexpr",
               "BD": "constexpr"},
    constexprs={"ROWS": ROWS, "B": B, "C": C, "D": D, "W": W, "BM": BM, "BA": BA, "BB": BB, "BC": BC,
                "BD": BD},
    args=[Arg("x", IN, "float32", (ROWS * W,)), Arg("o", INOUT, "float32", (BM * W,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="refuse",
    expect_why="an indirect transfer with no indirect_dim reads its "
               "tile-shaped index buffer from the start, so a peel's pieces would all read the "
               "first piece's indices",
)
