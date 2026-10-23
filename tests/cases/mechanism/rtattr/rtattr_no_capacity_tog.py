"""Pins R-CAP at tog, not at lower: a gather whose index the kernel computes from a read, in a loop
whose trip count is the scalar parameter iters, has no static row count for its snapshot. The ELF
runs unrecorded and passes; the TOG, which would read the snapshot, is refused by name. Breaks if
lower refuses (no ELF) or the TOG is built without the snapshot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

N, BLOCK, ITERS = 1024, 128, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_no_capacity_tog(x_ptr, idx_ptr, o_ptr, iters, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        kk = tl.load(idx_ptr + r)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for _ in range(iters):
            acc += tl.load(x_ptr + kk)
            kk += 1
        tl.store(o_ptr + r, acc)

    return rtattr_no_capacity_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "idx": torch.randint(0, N - ITERS, (BLOCK,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    x, idx = inputs["x"], inputs["idx"].long()
    return {"o": sum(x[idx + k] for k in range(ITERS)).to(torch.float32)}


SPEC = KernelSpec(
    name="rtattr_no_capacity_tog", kernel=kernel,
    signature={"x_ptr": "*fp32", "idx_ptr": "*i32", "o_ptr": "*fp32", "iters": "i32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("idx", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-4, atol=1e-4,
    extra={"scalar_args": [("iters", "int32_t")], "scalar_values": {"iters": ITERS}},
    expect="refuse", expect_stop=Stop("tog", Reason.RUNTIME_ATTRIBUTE),
    expect_why="the gather's index snapshot takes a row per trip of a loop the scalar iters bounds: "
               "no static capacity, so the ELF records nothing and the TOG that would read it is "
               "refused",
)
