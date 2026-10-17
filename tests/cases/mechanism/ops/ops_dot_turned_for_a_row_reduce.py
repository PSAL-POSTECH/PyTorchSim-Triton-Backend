"""A softmax over a dot's rows, 256 rows deep: the dot turned around so the rows sit on the lanes.

WHAT IT PINS. The array takes a dot's product with N on the lanes, and a reduce over the last
axis wants that axis off them, rows on the lanes. Left as it is, the [256, 256] score crosses
the lanes once for the row max and sum, and those two [256] statistics then gather back into the
second dot's layout to scale p and its product. Each dot here has M rows that fill the lanes as
well as its N columns and meets only row reduces, so anchor-layouts computes it as
C^T = B^T A^T: every operand and the product have their first axis on the lanes, where the
reduces keep it, and nothing moves between lanes.

    before   1 crossing, 2 gathers (torchsim.crosslane = 4, 5, 5), tiles [6, 3265] cycles
    after    no crosslane_move,                                    tiles [6, 2020] cycles,
             the same max_abs_err
    one walk the score's mask, row max and the rest each a walk of its own -> s and its max one
             walk, p, its bf16 copy and its row sum another: torchsim.vector_loop 7 -> 5,
             tiles [7, 1789] -> [7, 1712] cycles, the same max_abs_err

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
    def ops_dot_turned_for_a_row_reduce_kernel(
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

    return ops_dot_turned_for_a_row_reduce_kernel


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
    name="ops_dot_turned_for_a_row_reduce_kernel", kernel=kernel,
    signature={"a_ptr": "*bf16", "b_ptr": "*bf16", "v_ptr": "*bf16", "o_ptr": "*fp32",
               "R": "constexpr", "K": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "K": K, "C": C},
    args=[Arg("a", IN, "bfloat16", (R * K,)), Arg("b", IN, "bfloat16", (K * R,)),
          Arg("v", IN, "bfloat16", (R * C,)), Arg("o", OUT, "float32", (R, C))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=2e-2, atol=2e-2,
    # THE TURN IS THE CLAIM -- with both dots turned, no value moves between lanes.
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane_move": 0},
               "08-passes/10-torchsim-vectorize.mlir": {"torchsim.vector_loop": 5}},
)
