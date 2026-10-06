"""An unmasked store whose VALUE is a predicate: the bound is not the access's.

`x % W` may be dropped only where the access that reads it is masked to `x < W`.
This store has no mask at all -- the comparison is its DATA, widened to i16 --
so the wrap has to stay and every lane past W has to fold back into range.

torch equivalent: o[i % W] = int16(i < W), for i in range(BLOCK)

WHAT BREAKS IT: psto-drop-masked-modulo taking its bound from any operand whose type SPELLING contains "i1". The stored value is `tensor<64xi16>`, which matches, and peeling the widening reaches the comparison underneath -- so an unmasked store is read as masked by its own data and the wrap is dropped. 16 of 48 elements come back 1 where the reference says 0. The sibling is ptr_modulo_bounded_by_mask.py, the case the drop IS for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK, W = 64, 48       # 64 lanes wrap into 48 rows; 16 of them write twice


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_modulo_store_value_is_not_a_mask_kernel(o_ptr, W: tl.constexpr,
                                                    BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        # WIDENED ON PURPOSE. At i1 Triton extends a stored predicate to i8
        # before the store, and the defect below needs a width whose spelling
        # contains "i1".
        flag = (i < W).to(tl.int16)
        tl.store(o_ptr + (i % W), flag)          # NO MASK: every lane writes

    return ptr_modulo_store_value_is_not_a_mask_kernel


def make_inputs(torch, seed=0):
    return {}


def reference(inputs):
    import torch
    o = torch.zeros(W, dtype=torch.int16)
    for i in range(BLOCK):
        o[i % W] = 1 if i < W else 0             # the last write wins
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_modulo_store_value_is_not_a_mask_kernel",
    kernel=kernel,
    signature={"o_ptr": "*i16", "W": "constexpr", "BLOCK": "constexpr"},
    constexprs={"W": W, "BLOCK": BLOCK},
    args=[Arg("o", OUT, "int16", (W,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
