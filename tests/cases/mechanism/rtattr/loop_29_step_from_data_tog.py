"""Pins a loop whose step the kernel reads from memory as one the TOG runs by the recorded step: rows
0, s[0], 2 s[0], .. below NROWS of x are summed. Breaks if a step resting on a read is refused, or the
TOG runs the loop by another step."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, NROWS, STEP, G = 128, 7, 3, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_29_step_from_data_tog(x_ptr, s_ptr, o_ptr, NROWS: tl.constexpr, B: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, B)
        step = tl.load(s_ptr)
        acc = tl.zeros((B,), dtype=tl.float32)
        for t in range(0, NROWS, step):
            acc += tl.load(x_ptr + t * B + r)
        tl.store(o_ptr + pid * B + r, acc)

    return loop_29_step_from_data_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NROWS * B, generator=g, dtype=torch.float32),
            "s": torch.tensor([STEP], dtype=torch.int32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(NROWS, B)[::STEP].sum(0).repeat(G)}


SPEC = KernelSpec(
    name="loop_29_step_from_data_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "s_ptr": "*i32", "o_ptr": "*fp32", "NROWS": "constexpr", "B": "constexpr"},
    constexprs={"NROWS": NROWS, "B": B},
    args=[Arg("x", IN, "float32", (NROWS * B,)), Arg("s", IN, "int32", (1,)),
          Arg("o", OUT, "float32", (G * B,))],
    grid=(G,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
    expect_ir={"09-custom.mlir": {"torchsim.runtime_attribute": 1}},
)
