"""Several tensor tt.asserts of two ranks in one function, each reduced to its own scalar assert.

pytorchsim-reduce-tensor-assert builds one extui + tt.reduce chain per assert; the text edit it replaced gave
every chain the same block arguments, and GPT-2's lm test failed with 'redefinition of SSA value %a'. This kernel
is the shape of that module (a 1-D and two 2-D asserts); it did not fail the text edit on its own, which was
found only through GPT-2. Each holds at runtime, so the program runs to its end on Spike.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

N = 128
BLOCK = 128


def kernel():
    """Lazy: no triton import at module scope. See pytorchsim_triton_compiler/contract/kernel_spec.py."""
    import triton
    import triton.language as tl

    @triton.jit
    def assert_tensors_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)

        tl.device_assert(x == x, "no nans")
        y = tl.reshape(x, (8, 16))
        tl.device_assert(y == y, "no nans 2d")
        tl.device_assert(y < 1000.0, "small")

        tl.store(o_ptr + offs, x * 2.0)

    return assert_tensors_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"] * 2.0}


SPEC = KernelSpec(
    name="assert_tensors_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    extra={"ttir_options": {"debug": True}},
)
