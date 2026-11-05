"""`(r0_offset + r) % K` in a loop stepping by a multiple of K: the offset is whole laps and drops, leaving
a plain wrap on r, given r0_offset >= 0, read off the loop (lower bound 0, step > 0). Breaks if lowerBoundOf
stops reading an induction variable: the offset folds into the wrap and the load needs an index table."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, RB, K = 256, 64, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_rem_loop_offset_whole_laps_kernel(x_ptr, o_ptr, R: tl.constexpr, RB: tl.constexpr,
                                              K: tl.constexpr):
        acc = tl.zeros([RB], dtype=tl.float32)
        for r0_offset in range(0, R, RB):
            r = r0_offset + tl.arange(0, RB)
            acc += tl.load(x_ptr + r % K) * r.to(tl.float32)
        tl.store(o_ptr + tl.arange(0, RB), acc)

    return ptr_rem_loop_offset_whole_laps_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(R)
    return {"o": (inputs["x"][r % K] * r.to(torch.float32)).reshape(R // RB, RB).sum(0)}


SPEC = KernelSpec(
    name="ptr_rem_loop_offset_whole_laps_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "R": "constexpr", "RB": "constexpr", "K": "constexpr"},
    constexprs={"R": R, "RB": RB, "K": K},
    args=[Arg("x", IN, "float32", (K,)), Arg("o", OUT, "float32", (RB,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0}, "09-custom.mlir": {"pytorchsim.runtime_attribute": 0}},
)
