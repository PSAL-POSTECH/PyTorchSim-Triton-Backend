"""A rank-5 tile gathered along axis 1: peeling the outermost free axis would walk the gather axis,
so the peel takes axis 2 instead -- one MVIN per position, each reading its own index entries.
Breaks it: choosing the peeled axes first-fit (old tree: refused, 'gather axis 1 is peeled')."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS = 128
A, B, C, D = 8, 4, 4, 16
BA, BB, BC, BD = 2, 2, 2, 8
W = A * B * C * D


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_five_dim_peel_beside_gather_axis_kernel(
            x_ptr, i_ptr, o_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
            W: tl.constexpr, R: tl.constexpr, BA: tl.constexpr, BB: tl.constexpr,
            BC: tl.constexpr, BD: tl.constexpr):
        rm = tl.arange(0, R)
        ra = tl.arange(0, BA)
        rb = tl.arange(0, BB)
        rc = tl.arange(0, BC)
        rd = tl.arange(0, BD)
        idx = tl.load(i_ptr + ra)
        off = (rm[:, None, None, None, None] * W
               + idx[None, :, None, None, None] * (B * C * D)
               + rb[None, None, :, None, None] * (C * D)
               + rc[None, None, None, :, None] * D
               + rd[None, None, None, None, :])
        v = tl.load(x_ptr + off)
        oo = (rm[:, None, None, None, None] * (BA * BB * BC * BD)
              + ra[None, :, None, None, None] * (BB * BC * BD)
              + rb[None, None, :, None, None] * (BC * BD)
              + rc[None, None, None, :, None] * BD
              + rd[None, None, None, None, :])
        tl.store(o_ptr + oo, v * 2.0)

    return ptr_gather_five_dim_peel_beside_gather_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS * W, generator=g, dtype=torch.float32),
            "i": torch.randperm(A, generator=g)[:BA].to(torch.int32)}


def reference(inputs):
    x = inputs["x"].view(ROWS, A, B, C, D)[:, inputs["i"].long(), :BB, :BC, :BD]
    return {"o": (x * 2.0).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_gather_five_dim_peel_beside_gather_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32", "B": "constexpr",
               "C": "constexpr", "D": "constexpr", "W": "constexpr", "R": "constexpr",
               "BA": "constexpr", "BB": "constexpr", "BC": "constexpr", "BD": "constexpr"},
    constexprs={"B": B, "C": C, "D": D, "W": W, "R": ROWS, "BA": BA, "BB": BB, "BC": BC,
                "BD": BD},
    args=[Arg("x", IN, "float32", (ROWS * W,)), Arg("i", IN, "int32", (BA,)),
          Arg("o", OUT, "float32", (ROWS * BA * BB * BC * BD,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, expect_stage="timing",
)
