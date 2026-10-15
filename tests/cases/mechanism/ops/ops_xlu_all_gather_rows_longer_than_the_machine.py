"""A [256] row statistic made in one dot's layout and read in another's: a gather whose source
axis is two lane-fulls.

WHAT IT PINS. The row sum of a softmax is computed where the score tile lives and divides the
second dot's [256, 128] result, so every lane needs rows it does not hold. `gather_rounds` used
to refuse a source axis 0 longer than the lanes, and the move fell to an element-wise PERMUTE
(one pop per value). Folded onto the lanes, a pass still carries one value per lane, so it is a
gather of one more pass per lane-full.

    before   2 permutes (torchsim.crosslane = 16), tile 15297 cycles
    after    2 gathers  (torchsim.crosslane = 5),  tile 3265 cycles, the same output bits

At 128 rows the source is one lane-full and both trees already gather.

torch equivalent: softmax(a @ b) @ v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, K, C = 256, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_all_gather_rows_longer_than_the_machine_kernel(
            a_ptr, b_ptr, v_ptr, o_ptr, R: tl.constexpr, K: tl.constexpr, C: tl.constexpr):
        r = tl.arange(0, R)
        k = tl.arange(0, K)
        c = tl.arange(0, C)
        a = tl.load(a_ptr + r[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * R + r[None, :])
        v = tl.load(v_ptr + r[:, None] * C + c[None, :])
        s = tl.dot(a, b, out_dtype=tl.float32)
        p = tl.exp(s - tl.max(s, axis=1)[:, None])
        l = tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        tl.store(o_ptr + r[:, None] * C + c[None, :], acc / l[:, None])

    return ops_xlu_all_gather_rows_longer_than_the_machine_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(R * K, generator=g).to(torch.bfloat16),
            "b": torch.randn(K * R, generator=g).to(torch.bfloat16),
            "v": torch.randn(R * C, generator=g).to(torch.bfloat16)}


def reference(inputs):
    import torch
    s = inputs["a"].float().reshape(R, K) @ inputs["b"].float().reshape(K, R)
    return {"o": torch.softmax(s, -1) @ inputs["v"].float().reshape(R, C)}


SPEC = KernelSpec(
    name="ops_xlu_all_gather_rows_longer_than_the_machine_kernel", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "v_ptr": "*bf16", "o_ptr": "*fp32",
               "R": "constexpr", "K": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "K": K, "C": C},
    args=[Arg("a", IN, "bfloat16", (R * K,)), Arg("b", IN, "bfloat16", (K * R,)),
          Arg("v", IN, "bfloat16", (R * C,)), Arg("o", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    # THE MOVE IS THE CLAIM -- the row max and row sum reach the second dot's layout by gather.
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane = 5 : i64": 2,
                                       "torchsim.crosslane = 16 : i64": 0}},
)
