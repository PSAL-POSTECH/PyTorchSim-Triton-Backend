"""An element-masked store whose repeating axis is the LANE axis and whose last lap is partial.

    col = cn % 8, cn = 0..11          columns 8..11 alias 0..3: laps of 8 and 4
    keep = ((rm + cn) % 3) != 0       an element mask, no per-axis box
    o[rm, col[cn]] = x[rm, cn]        under keep, last write winning

WHAT BREAKS IT: one whole-tile read-modify-write puts a masked-off column 8..11 back over the live write of its
twin 0..3; the store has to be cut into laps (8 + 4 columns) issued in order, each a box read-modify-write.
Refused by name before laps existed (MASKED_STORE_NOT_A_BOX).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, BN, W = 4, 12, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_repeating_col_partial_lap_kernel(x_ptr, o_ptr, BM: tl.constexpr,
                                              BN: tl.constexpr, W: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, 16)
        keep = (((rm[:, None] + rn[None, :]) % 3) != 0) & (rn[None, :] < BN)
        v = tl.load(x_ptr + rm[:, None] * 16 + rn[None, :])
        tl.store(o_ptr + rm[:, None] * W + (rn % W)[None, :], v, mask=keep)

    return mask_repeating_col_partial_lap_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM * 16, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(BM, 16)
    o = torch.zeros(BM, W, dtype=torch.float32)
    for m in range(BM):
        for n in range(BN):
            if (m + n) % 3 != 0:
                o[m, n % W] = x[m, n]
    return {"o": o.reshape(-1)}


SPEC = KernelSpec(
    name="mask_repeating_col_partial_lap_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BM": "constexpr",
               "BN": "constexpr", "W": "constexpr"},
    constexprs={"BM": BM, "BN": BN, "W": W},
    args=[Arg("x", IN, "float32", (BM * 16,)),
          Arg("o", OUT, "float32", (BM * W,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="pass",
)
