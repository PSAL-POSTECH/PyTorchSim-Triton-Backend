"""A linalg.generic that yields TWO values, and the store that only made one."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_umulhi_kernel(n_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        n = tl.load(n_ptr + r)
        tl.store(o_ptr + r, tl.umulhi(n, n))

    return ops_umulhi_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"n": torch.randint(1 << 20, 1 << 30, (N,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    # In int64 so the product is exact, then take the top word. int64 rather
    # than uint32 arithmetic because the inputs are positive and below 2^30, so
    # signed and unsigned readings agree and int64 cannot overflow.
    w = inputs["n"].to(torch.int64) * inputs["n"].to(torch.int64)
    return {"o": (w >> 32).to(torch.int32).contiguous()}


SPEC = KernelSpec(
    name="ops_umulhi_kernel",
    kernel=kernel,
    signature={"n_ptr": "*i32", "o_ptr": "*i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("n", IN, "int32", (N,)), Arg("o", OUT, "int32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
