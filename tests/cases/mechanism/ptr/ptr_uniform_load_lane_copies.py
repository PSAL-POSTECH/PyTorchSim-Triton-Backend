"""A [R] row at a runtime stride, a UNIFORM [1, R] buffer copied to every lane: pins togcheck's 2b, which this
commit adds (92028c8 alone passes). Old emitter + this checker: "the TOG states 8 elements ... walks that many:
[1024]"; old emitter + new ir_transfers: "the IR has a transfer the TOG never emits: dims=[128, 1, 1, 8]"."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, R, SC = 128, 8, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_uniform_load_lane_copies_kernel(c_ptr, w_ptr, o_ptr, sc,
                                            X: tl.constexpr, R: tl.constexpr):
        r = tl.arange(0, R)
        x = tl.arange(0, X)[:, None]
        c = tl.load(c_ptr + r * sc)
        tl.store(o_ptr + x * R + r[None, :], tl.load(w_ptr + x * R + r[None, :]) * c[None, :])

    return ptr_uniform_load_lane_copies_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"c": torch.randn(R * SC, generator=g, dtype=torch.float32),
            "w": torch.randn(X * R, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": (inputs["w"].view(X, R) * inputs["c"][::SC][None, :]).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_uniform_load_lane_copies_kernel", kernel=kernel,
    signature={"c_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32", "sc": "i32",
               "X": "constexpr", "R": "constexpr"},
    constexprs={"X": X, "R": R},
    args=[Arg("c", IN, "float32", (R * SC,)), Arg("w", IN, "float32", (X * R,)),
          Arg("o", OUT, "float32", (X * R,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=0, atol=0,
    extra={"scalar_args": [("sc", "int32_t")], "scalar_values": {"sc": SC}},
    expect_stage="togcheck",
)
