"""Two rank-5 row gathers whose tiles peel: the TOG must name one DMA per hardware MVIN, so each
reads its own index dump. Pins the timing stage. Breaks it: peeling after the IR the TOG is built
from (old tree: TOGSim refused the 5-dim tile the TOG still named)."""
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
    def ptr_gather_five_dim_peeled_kernel(x_ptr, y_ptr, i_ptr, o_ptr,
                                          B: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
                                          W: tl.constexpr, BM: tl.constexpr, BA: tl.constexpr,
                                          BB: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr):
        rm = tl.arange(0, BM)
        ra = tl.arange(0, BA)
        rb = tl.arange(0, BB)
        rc = tl.arange(0, BC)
        rd = tl.arange(0, BD)
        idx = tl.load(i_ptr + rm)
        inner = (ra[None, :, None, None, None] * (B * C * D)
                 + rb[None, None, :, None, None] * (C * D)
                 + rc[None, None, None, :, None] * D
                 + rd[None, None, None, None, :])
        v = tl.load(x_ptr + idx[:, None, None, None, None] * W + inner)
        u = tl.load(y_ptr + idx[:, None, None, None, None] * W + inner)
        tl.store(o_ptr + rm[:, None, None, None, None] * W + inner, v + u)

    return ptr_gather_five_dim_peeled_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS * W, generator=g, dtype=torch.float32),
            "y": torch.randn(ROWS * W, generator=g, dtype=torch.float32),
            "i": torch.randperm(ROWS, generator=g)[:BM].to(torch.int32),
            "o": torch.zeros(BM * W, dtype=torch.float32)}


def reference(inputs):
    i = inputs["i"].long()
    s = (inputs["x"].view(ROWS, A, B, C, D)[i] + inputs["y"].view(ROWS, A, B, C, D)[i])
    o = inputs["o"].clone().view(BM, A, B, C, D)
    o[:, :BA, :BB, :BC, :BD] = s[:, :BA, :BB, :BC, :BD]
    return {"o": o.reshape(-1)}


SPEC = KernelSpec(
    name="ptr_gather_five_dim_peeled_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "B": "constexpr", "C": "constexpr", "D": "constexpr", "W": "constexpr",
               "BM": "constexpr", "BA": "constexpr", "BB": "constexpr", "BC": "constexpr",
               "BD": "constexpr"},
    constexprs={"B": B, "C": C, "D": D, "W": W, "BM": BM, "BA": BA, "BB": BB, "BC": BC,
                "BD": BD},
    args=[Arg("x", IN, "float32", (ROWS * W,)), Arg("y", IN, "float32", (ROWS * W,)),
          Arg("i", IN, "int32", (BM,)), Arg("o", INOUT, "float32", (BM * W,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_stage="timing",
)
