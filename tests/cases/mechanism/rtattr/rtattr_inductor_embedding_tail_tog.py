"""Pins an index snapshot of the leading entries of a longer index tile: Inductor's embedding lookup
masks a block past xnumel (56 tokens x 16 of a 1024 block), so the gather moves 896 elements and its
index buffer holds 1024. The snapshot records the 896 the gather reads. Breaks if the kernel is
refused at lower ("no one entry per position") or the offsets differ from the model's index dump."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

V, D, T, XBLOCK = 32, 16, 56, 1024
IDS = [(11 * k + 5) % V for k in range(T)]


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_inductor_embedding_tail_tog(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr):
        xnumel = 896
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = xindex // 16
        x0 = xindex % 16
        tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
        tmp2 = tl.load(in_ptr1 + (x0 + 16 * tmp0), xmask)
        tl.store(out_ptr0 + xindex, tmp2, xmask)

    return rtattr_inductor_embedding_tail_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"ids": torch.tensor(IDS, dtype=torch.int32), "w": torch.randn(V * D, generator=g)}


def reference(inputs):
    w = inputs["w"].reshape(V, D)
    return {"o": w[[int(i) for i in IDS]].reshape(-1)}


SPEC = KernelSpec(
    name="rtattr_inductor_embedding_tail_tog", kernel=kernel,
    signature={"in_ptr0": "*i32", "in_ptr1": "*fp32", "out_ptr0": "*fp32", "XBLOCK": "constexpr"},
    constexprs={"XBLOCK": XBLOCK},
    args=[Arg("ids", IN, "int32", (T,)), Arg("w", IN, "float32", (V * D,)), Arg("o", OUT, "float32", (T * D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
)
