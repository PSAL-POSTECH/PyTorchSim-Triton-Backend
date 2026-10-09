"""A modulo the access's own mask already bounds away: `x % W` where `x < W`.

WHAT IT PINS: ttclean's _drop_masked_modulo, and the contract it leans on. The lanes that would
wrap are exactly the lanes the mask turns off, and a masked-off position is
parked in the buffer's padding (wrapper allocates every buffer doubled),
so its address never had to be in range. The bound has to come from the mask of
EVERY access that reads the modulo -- taking it from any comparison in the
function is what made the first version of _drop_masked_modulo return wrong numbers.

WHAT PROVES IT is the descriptor, not the op -- a wrap stays a `tts.load` and
becomes an index buffer one stage later:
    on   dram_stride [1]              no indirect, no remsi
    off  dram_stride [0], indirect    one remsi feeding an index buffer

_drop_masked_modulo: the only one of the four that leans on the parking contract rather than on arithmetic alone.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

W, BLOCK = 48, 64      # the block stages 64 rows; only 48 exist


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_modulo_bounded_by_mask_kernel(x_ptr, o_ptr, W: tl.constexpr,
                                          BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        # `i % W` keeps the address in range; `i < W` says those lanes are dead
        # anyway, so the remainder is the identity on every lane that counts.
        v = tl.load(x_ptr + (i % W), i < W, other=0.0)
        tl.store(o_ptr + i, v)

    return ptr_modulo_bounded_by_mask_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(W, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(BLOCK)
    x = torch.cat([inputs["x"], torch.zeros(BLOCK - W)])
    return {"o": torch.where(i < W, x, torch.zeros(BLOCK))}


SPEC = KernelSpec(
    name="ptr_modulo_bounded_by_mask_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "W": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"W": W, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (W,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"09-custom.mlir": {"arith.remsi": 0, "offset_stride": 0}},
)
