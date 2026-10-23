"""Pins a DMA in an scf.while whose trip count is a kernel scalar as a loop the TOG runs to the
condition it computes: `while t < ntrips` sums ntrips rows of x. Breaks if the TOG refuses a loop it
cannot count before running it, or runs a fixed number of trips."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, NTRIPS, G = 128, 5, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_27_dma_in_while_tog(x_ptr, o_ptr, ntrips, B: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, B)
        acc = tl.zeros((B,), dtype=tl.float32)
        t = 0
        while t < ntrips:
            acc += tl.load(x_ptr + t * B + r)
            t += 1
        tl.store(o_ptr + pid * B + r, acc)

    return loop_27_dma_in_while_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NTRIPS * B, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(NTRIPS, B).sum(0).repeat(G)}


SPEC = KernelSpec(
    name="loop_27_dma_in_while_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "ntrips": "i32", "B": "constexpr"},
    constexprs={"B": B},
    args=[Arg("x", IN, "float32", (NTRIPS * B,)), Arg("o", OUT, "float32", (G * B,))],
    grid=(G,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
    extra={"scalar_args": [("ntrips", "int32_t")], "scalar_values": {"ntrips": NTRIPS}},
)
