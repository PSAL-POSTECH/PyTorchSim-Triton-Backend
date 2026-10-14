"""The write of step_split_tile_under_an_elem_mask: rows `(m // K) * SH + (m % K) * SW` under a mask from memory.

The value is expanded onto the split tile [BM/K, K, BC], and tts.store wants its elem_mask in the
value's shape, so the [BM, BC] mask has to be expanded the same way.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, BC, K, SH, SW = 64, 16, 4, 256, 32
NO = (BM // K - 1) * SH + (K - 1) * SW + BC


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def step_split_tile_under_an_elem_mask_store_kernel(a_ptr, f_ptr, o_ptr, BM: tl.constexpr,
                                                        BC: tl.constexpr, K: tl.constexpr,
                                                        SH: tl.constexpr, SW: tl.constexpr):
        m = tl.arange(0, BM)
        c = tl.arange(0, BC)
        keep = tl.load(f_ptr + m) != 0
        v = tl.load(a_ptr + m[:, None] * BC + c[None, :])
        row = (m // K) * SH + (m % K) * SW
        tl.store(o_ptr + row[:, None] + c[None, :], v, mask=keep[:, None])

    return step_split_tile_under_an_elem_mask_store_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(BM * BC, generator=g, dtype=torch.float32),
            "f": (torch.arange(BM, dtype=torch.int32) % 3).contiguous()}


def reference(inputs):
    import torch
    m = torch.arange(BM)
    row = (m // K) * SH + (m % K) * SW
    keep = inputs["f"] != 0
    o = torch.zeros(NO, dtype=torch.float32)
    idx = (row[:, None] + torch.arange(BC)[None, :])[keep]
    o[idx.reshape(-1)] = inputs["a"].reshape(BM, BC)[keep].reshape(-1)
    return {"o": o}


SPEC = KernelSpec(
    name="step_split_tile_under_an_elem_mask_store_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "f_ptr": "*i32", "o_ptr": "*fp32", "BM": "constexpr",
               "BC": "constexpr", "K": "constexpr", "SH": "constexpr", "SW": "constexpr"},
    constexprs={"BM": BM, "BC": BC, "K": K, "SH": SH, "SW": SW},
    args=[Arg("a", IN, "float32", (BM * BC,)), Arg("f", IN, "int32", (BM,)),
          Arg("o", OUT, "float32", (NO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
