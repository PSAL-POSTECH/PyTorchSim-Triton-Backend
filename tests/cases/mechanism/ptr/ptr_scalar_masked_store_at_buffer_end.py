"""A masked scalar store into a 512-byte output: the masked-off write must land nowhere, not one element past the buffer.
Each masked store is a one-element transfer bounded by its mask, [0, mask ? 1 : 0): two masked transfers and
no scf.if, which a TOG cannot state."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_scalar_masked_store_at_buffer_end_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        tl.store(o_ptr + r, tl.load(x_ptr + r))
        v = tl.load(x_ptr + 3, mask=n > 0, other=0.0)
        tl.store(o_ptr + 5, v, mask=n > 0)      # taken
        tl.store(o_ptr + 7, v, mask=n < 0)      # NOT taken: o[7] keeps x[7]

    return ptr_scalar_masked_store_at_buffer_end_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    o = inputs["x"].clone()
    o[5] = inputs["x"][3]
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_scalar_masked_store_at_buffer_end_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t")], "scalar_values": {"n": N}},
    expect_ir={"08-transformed.mlir": {"scf.if": 0, "masked_axes": 2}},
)
