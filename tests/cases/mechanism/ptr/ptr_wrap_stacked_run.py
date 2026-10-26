"""A stacked wraparound that actually RUNS: rows carry `% WM`.

SIBLING: ptr_wrap_stacked.py is the same shape with a RUNTIME start, a
runtime extent and a mask -- the three ways _cut_facts refuses. This one
has none of them, so it is the one that gets its runs.

HOW IT IS LOWERED: ONE PIECE PER LAP

`shape[0] = W` says the dim-0 contribution is taken modulo W (already scaled by
the dim's stride, so `% 12` on a stride-32 row arrives as 384). Within one lap
the address is `o + p*st - j*W`, linear in p, so each lap is an ordinary
strided transfer -- and a wrapped access is exactly product-of-laps many of
them. a02 states the runs and c18 stamps one descriptor per piece:

    pytorchsim.cut = [array<i64: 0, 32, 384, 0>, array<i64>]

    One (radix, stride, modulus, phase) digit record per digit, one array per
    tile axis and empty where the axis is not cut. Axis 0 is one wrapping digit
    of stride 32, modulus 384, phase 0, which c18 splits into two runs: rows
    0..11 read straight through (delta 0), rows 12..15 restart at 0 -- the same
    walk shifted back by 384. Axis 1 is uncut.

THIS USED TO BE AN INDIRECT TRANSFER, and the reason it no longer is belongs
here: two pieces writing into one scratchpad buffer used to overwrite each
other's rows with masked_fill, because the descriptor could not say "leave the
rest alone". It can now -- F_SKIP_AXIS, one bit per axis, writes NOTHING
outside [lo, hi) instead of the fill. The indirect form is still what a wrap
falls back to when the laps cannot be counted here; ptr_wrap_stacked.py is the
three ways that happens.

minimal stacked wrap
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
WM, BM, BN = 12, 16, 32          # 12 < 16 -> the seam falls inside the tile
NX = 64 * BN
def kernel():
    import triton, triton.language as tl
    @triton.jit
    def ptr_wrap_stacked_run_kernel(x_ptr, o_ptr, WM: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM); rn = tl.arange(0, BN)
        rows = rm % WM
        v = tl.load(x_ptr + rows[:, None] * BN + rn[None, :])
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], v)
    return ptr_wrap_stacked_run_kernel
def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}
def reference(inputs):
    import torch
    rm = torch.arange(BM); rn = torch.arange(BN)
    return {"o": inputs["x"][(rm % WM)[:, None] * BN + rn[None, :]].reshape(-1)}
SPEC = KernelSpec(name="ptr_wrap_stacked_run_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "WM": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"WM": WM, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs, )
