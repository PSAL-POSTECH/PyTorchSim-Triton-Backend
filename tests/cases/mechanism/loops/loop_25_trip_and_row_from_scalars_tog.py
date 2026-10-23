"""Pins kernel scalars reaching the TOG: a loop runs `ntrips` times from row `row0`, both i32 parameters,
after a float and an unused i32 the TOG must skip by position. Breaks if the TOG is refused a scalar shape_args
does not carry, or binds one by the wrong position (row0 read as `pad` reaches past x's 8 rows)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, ROW0, NTRIPS, PAD, ALPHA, G = 128, 3, 5, 1000, 0.5, 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_25_trip_and_row_from_scalars_tog(x_ptr, o_ptr, alpha, pad, row0, ntrips, B: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, B)
        acc = tl.zeros((B,), dtype=tl.float32)
        for t in range(0, ntrips):
            acc += tl.load(x_ptr + (row0 + t) * B + r)
        tl.store(o_ptr + pid * B + r, acc * alpha)

    return loop_25_trip_and_row_from_scalars_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn((ROW0 + NTRIPS) * B, generator=g, dtype=torch.float32)}


def reference(inputs):
    s = inputs["x"].reshape(ROW0 + NTRIPS, B)[ROW0:].sum(0) * ALPHA
    return {"o": s.repeat(G)}


SPEC = KernelSpec(
    name="loop_25_trip_and_row_from_scalars_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "alpha": "fp32", "pad": "i32", "row0": "i32",
               "ntrips": "i32", "B": "constexpr"},
    constexprs={"B": B},
    args=[Arg("x", IN, "float32", ((ROW0 + NTRIPS) * B,)), Arg("o", OUT, "float32", (G * B,))],
    grid=(G,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
    extra={"scalar_args": [("alpha", "float"), ("pad", "int32_t"), ("row0", "int32_t"), ("ntrips", "int32_t")],
           "scalar_values": {"alpha": ALPHA, "pad": PAD, "row0": ROW0, "ntrips": NTRIPS}},
)
