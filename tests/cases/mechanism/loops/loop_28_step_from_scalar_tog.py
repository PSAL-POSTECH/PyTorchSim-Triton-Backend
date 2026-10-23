"""Pins a DMA in an scf.for whose step is a kernel scalar as a loop the TOG runs by that step: rows
0, step, 2 step, .. below nrows of x are summed. Breaks if the TOG refuses a loop whose step is no
constant, or runs it by another step."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, NROWS, STEP, G = 128, 7, 3, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_28_step_from_scalar_tog(x_ptr, o_ptr, nrows, step, B: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, B)
        acc = tl.zeros((B,), dtype=tl.float32)
        for t in range(0, nrows, step):
            acc += tl.load(x_ptr + t * B + r)
        tl.store(o_ptr + pid * B + r, acc)

    return loop_28_step_from_scalar_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NROWS * B, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(NROWS, B)[::STEP].sum(0).repeat(G)}


SPEC = KernelSpec(
    name="loop_28_step_from_scalar_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "nrows": "i32", "step": "i32", "B": "constexpr"},
    constexprs={"B": B},
    args=[Arg("x", IN, "float32", (NROWS * B,)), Arg("o", OUT, "float32", (G * B,))],
    grid=(G,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
    extra={"scalar_args": [("nrows", "int32_t"), ("step", "int32_t")],
           "scalar_values": {"nrows": NROWS, "step": STEP}},
)
