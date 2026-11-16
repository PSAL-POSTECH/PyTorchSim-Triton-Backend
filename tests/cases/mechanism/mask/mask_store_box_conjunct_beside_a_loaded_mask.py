"""A store under `(r < K) & (r < bnd)`, bnd read from memory: the `r < K` conjunct stays the store's box.

MaskAnalysis cannot parse the whole mask (the loaded bound), so the whole mask rides as the element mask; the
conjunct it can state still bounds the box, and the read-modify-write covers K lanes, not the tile. Breaks if
the box falls back to the full extent (static_mask_dims 256 on the store) or a masked-off lane is written.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, INOUT

BLOCK, K, BNDVAL = 256, 128, 100


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_store_box_conjunct_beside_a_loaded_mask_kernel(x_ptr, b_ptr, o_ptr, BLOCK: tl.constexpr,
                                                           K: tl.constexpr):
        r = tl.arange(0, BLOCK)
        bnd = tl.load(b_ptr + r)
        v = tl.load(x_ptr + r)
        tl.store(o_ptr + r, v, mask=(r < K) & (r < bnd))

    return mask_store_box_conjunct_beside_a_loaded_mask_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BLOCK, generator=g, dtype=torch.float32),
            "b": torch.full((BLOCK,), BNDVAL, dtype=torch.int32),
            "o": torch.full((BLOCK,), 7.0, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(BLOCK)
    return {"o": torch.where(r < min(K, BNDVAL), inputs["x"], inputs["o"])}


SPEC = KernelSpec(
    name="mask_store_box_conjunct_beside_a_loaded_mask_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "b_ptr": "*i32", "o_ptr": "*fp32", "BLOCK": "constexpr", "K": "constexpr"},
    constexprs={"BLOCK": BLOCK, "K": K},
    args=[Arg("x", IN, "float32", (BLOCK,)), Arg("b", IN, "int32", (BLOCK,)),
          Arg("o", INOUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.scatter": 0, "static_mask_dims = array<i64: 128>": 1}},
)
