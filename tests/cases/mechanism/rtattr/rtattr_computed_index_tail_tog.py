"""Pins the snapshot of a computed-index gather (no indirect_dim) whose index tile is longer than the moved
tile: MobileNet's per-channel `in_ptr1 + x % 32` under a 512 block of 384 elements moves [384] and reads a
[512] index. Breaks if lower refuses it ("no one entry per position") or the snapshot is not x % 32."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, XBLOCK, C = 384, 512, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_computed_index_tail_tog(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr):
        xnumel = 384
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex % 32
        tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + xindex, xmask)
        tl.store(out_ptr0 + xindex, tmp0 + tmp1, xmask)

    return rtattr_computed_index_tail_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"b": torch.randn(C, generator=g), "x": torch.randn(N, generator=g)}


def reference(inputs):
    return {"o": inputs["x"] + inputs["b"].repeat(N // C)}


SPEC = KernelSpec(
    name="rtattr_computed_index_tail_tog", kernel=kernel,
    signature={"in_ptr0": "*fp32", "in_ptr1": "*fp32", "out_ptr0": "*fp32", "XBLOCK": "constexpr"},
    constexprs={"XBLOCK": XBLOCK},
    args=[Arg("b", IN, "float32", (C,)), Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
    expect_ir={"09-custom.mlir": {"pytorchsim.runtime_attribute": 1}},
)
