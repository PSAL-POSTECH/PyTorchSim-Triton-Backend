"""Pins DMAs in both arms of an scf.if that yields the tile as the TOG's if/else: even programs load
a, odd ones b and double it, so program 0 reads a's row and program 1 b's. Breaks if the TOG refuses a
branch whose else does work or that yields, or states one arm for every program."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, G = 128, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_26_dma_in_both_arms_tog(a_ptr, b_ptr, o_ptr, B: tl.constexpr):
        pid = tl.program_id(0)
        r = pid * B + tl.arange(0, B)
        if pid % 2 == 0:
            x = tl.load(a_ptr + r)
        else:
            x = tl.load(b_ptr + r) * 2.0
        tl.store(o_ptr + r, x)

    return loop_26_dma_in_both_arms_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {x: torch.randn(G * B, generator=g, dtype=torch.float32) for x in "ab"}


def reference(inputs):
    a, b = (inputs[x].reshape(G, B) for x in "ab")
    o = a.clone()
    o[1::2] = b[1::2] * 2.0
    return {"o": o.reshape(-1)}


SPEC = KernelSpec(
    name="loop_26_dma_in_both_arms_tog", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32", "B": "constexpr"},
    constexprs={"B": B},
    args=[Arg("a", IN, "float32", (G * B,)), Arg("b", IN, "float32", (G * B,)),
          Arg("o", OUT, "float32", (G * B,))],
    grid=(G,), reference=reference, make_inputs=make_inputs, rtol=1e-6, atol=1e-6,
)
