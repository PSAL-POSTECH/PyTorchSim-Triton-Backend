"""Inductor's slice_backward over a cat: three loads whose masks bound `x // K` from both sides, each at an
offset that is negative where its mask is off (x - 96, x - 224). triton_poi_fused_add_mm_slice_backward_9 of
PyTorchSim's test_grouped_mm, at a smaller shape.

WHAT BREAKS IT: MaskAnalysis not reading `q >= c`, `q < c` through `divsi` and the `extsi` after it. The
masks stay whole-tile element masks, the loads become full-extent boxes starting before their tensor, and
simplify-index-arithmetic seeded each box's DRAM offset as inside its argument: [-96, -96] met [0, 127] in
no value and its worklist narrowed forever (the compile never ended).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, BLOCK = 32, 256
A, B, C = 96, 128, 32           # rows 0..2, 3..6, 7 of K
N = A + B + C


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_quotient_ranges_on_slice_offsets_kernel(
            a_ptr, b_ptr, c_ptr, o_ptr, K: tl.constexpr, BLOCK: tl.constexpr, N: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        q = (x // K).to(tl.int64)
        inside = x < N
        va = tl.load(a_ptr + x, (q < 3) & inside, other=0.0)
        vb = tl.load(b_ptr + (x - 96), (q >= 3) & (q < 7) & inside, other=0.0)
        vc = tl.load(c_ptr + (x - 224), (q >= 7) & inside, other=0.0)
        tl.store(o_ptr + x, va + vb + vc, inside)

    return mask_quotient_ranges_on_slice_offsets_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(A, generator=g, dtype=torch.float32),
            "b": torch.randn(B, generator=g, dtype=torch.float32),
            "c": torch.randn(C, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": torch.cat([inputs["a"], inputs["b"], inputs["c"]])}


SPEC = KernelSpec(
    name="mask_quotient_ranges_on_slice_offsets_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "o_ptr": "*fp32",
               "K": "constexpr", "BLOCK": "constexpr", "N": "constexpr"},
    constexprs={"K": K, "BLOCK": BLOCK, "N": N},
    args=[Arg("a", IN, "float32", (A,)), Arg("b", IN, "float32", (B,)), Arg("c", IN, "float32", (C,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0, "static_mask_lows": 2}},
)
