"""Inductor's split GEMV: `_acc = _acc + x * b` over K chunks into an [N, R] tile, `tl.sum(_acc, 1)`
after the loop. The loop carries [N] and sums each chunk in it, so no [N, R] f32 tile is a buffer.
Breaks if tritonpytorchsim-reduce-in-loop stops matching the carried-combine-then-reduce shape."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, N, R = 128, 256, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sum_after_accumulate_loop_kernel(a_ptr, b_ptr, c_ptr):
        xindex = tl.arange(0, 256)[:, None]
        r0_base = tl.arange(0, 32)[None, :]
        _acc = tl.full([256, 32], 0, tl.float32)
        for r0_offset in range(0, 128, 32):
            r0 = r0_offset + r0_base
            x = tl.load(a_ptr + r0).to(tl.float32)
            b = tl.load(b_ptr + (xindex + 256 * r0)).to(tl.float32)
            _acc = _acc + tl.broadcast_to(x * b, [256, 32])
        y = tl.sum(_acc, 1)[:, None]
        tl.store(c_ptr + xindex, y)

    return sum_after_accumulate_loop_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(1, K, generator=g, dtype=torch.float32).to(torch.bfloat16),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.bfloat16),
    }


def reference(inputs):
    import torch
    return {"c": torch.matmul(inputs["a"].to(torch.float32),
                              inputs["b"].to(torch.float32)).to(torch.bfloat16)}


SPEC = KernelSpec(
    name="sum_after_accumulate_loop_kernel",
    kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16"},
    constexprs={},
    args=[Arg("a", IN, "bfloat16", (1, K)), Arg("b", IN, "bfloat16", (K, N)),
          Arg("c", OUT, "bfloat16", (1, N))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
    expect_ir={"08-transformed.mlir": {"memref<256x32xf32, 1>": 0}},
)
