"""Pins kernel scalars that sit between pointers in the signature (x_ptr, ntrips, o_ptr, row0) reaching
the TOG: a loop runs `ntrips` times from row `row0`. Breaks if the TOG takes a scalar's argument as its
place among the scalars after every pointer (ntrips read as o_ptr's argument, row0 as past the end)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, ROW0, NTRIPS, G = 128, 2, 3, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_scalar_between_pointers_tog(x_ptr, ntrips, o_ptr, row0, B: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, B)
        acc = tl.zeros((B,), dtype=tl.float32)
        for t in range(0, ntrips):
            acc += tl.load(x_ptr + (row0 + t) * B + r)
        tl.store(o_ptr + pid * B + r, acc)

    return rtattr_scalar_between_pointers_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn((ROW0 + NTRIPS) * B, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].reshape(ROW0 + NTRIPS, B)[ROW0:].sum(0).repeat(G)}


SPEC = KernelSpec(
    name="rtattr_scalar_between_pointers_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "ntrips": "i32", "o_ptr": "*fp32", "row0": "i32", "B": "constexpr"},
    constexprs={"B": B},
    args=[Arg("x", IN, "float32", ((ROW0 + NTRIPS) * B,)), Arg("o", OUT, "float32", (G * B,))],
    grid=(G,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
    extra={"scalar_args": [("ntrips", "int32_t"), ("row0", "int32_t")],
           "scalar_values": {"ntrips": NTRIPS, "row0": ROW0}},
)
