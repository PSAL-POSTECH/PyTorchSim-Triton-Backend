"""The store mask that conv15 builds, dumped as a value instead of used.

resnet18's sixteenth conv masks its store with

    idx_n = (nhw // OUT_W) // OUT_H ;  keep = idx_n < BATCH

and MEASURED on hardware that mask is true for nhw = 49..55, where idx_n is 1
and it must be false. The same idx_n reaches the ADDRESS path correctly -- the
gather offsets for those rows carry idx_n * stride_xn = 50176 -- so the divisions
are right and something between the compare and the mask's use is not.

WHAT THIS KERNEL SEPARATES. It writes the predicate itself, so a wrong answer is
read directly rather than inferred from which output rows went missing. The
second axis is here because the failing mask is a 64-element i1 broadcast to
[64, 128], which the backend spills to DRAM scratch and reads back with a zero
stride -- a step a one-dimensional mask never takes.

Dumps the nested-division store mask conv15 builds, to read the predicate directly rather than infer it from missing output rows.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, N, OUT_W, OUT_H, BATCH = 64, 128, 7, 7, 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_from_nested_div_kernel(o_ptr, BM: tl.constexpr, N: tl.constexpr,
                                    OUT_W: tl.constexpr, OUT_H: tl.constexpr,
                                    BATCH: tl.constexpr):
        nhw = tl.arange(0, BM)
        rn = tl.arange(0, N)
        idx_y_w = nhw % OUT_W
        nh = nhw // OUT_W
        idx_y_h = nh % OUT_H
        idx_n = nh // OUT_H
        keep = ((idx_n < BATCH) & (idx_y_h < OUT_H) & (idx_y_w < OUT_W))[:, None]
        v = tl.where(keep & (rn[None, :] < N), 1.0, 0.0)
        tl.store(o_ptr + nhw[:, None] * N + rn[None, :], v)

    return mask_from_nested_div_kernel


def make_inputs(torch, seed=0):
    return {}


def reference(inputs):
    import torch
    o = torch.zeros(BM, N, dtype=torch.float32)
    for m in range(BM):
        if (m // OUT_W) // OUT_H < BATCH:
            o[m] = 1.0
    return {"o": o.reshape(-1)}


SPEC = KernelSpec(
    name="mask_from_nested_div_kernel", kernel=kernel,
    signature={"o_ptr": "*fp32", "BM": "constexpr", "N": "constexpr",
               "OUT_W": "constexpr", "OUT_H": "constexpr",
               "BATCH": "constexpr"},
    constexprs={"BM": BM, "N": N, "OUT_W": OUT_W, "OUT_H": OUT_H,
                "BATCH": BATCH},
    args=[Arg("o", OUT, "float32", (BM * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
