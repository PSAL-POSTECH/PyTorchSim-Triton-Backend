"""Qwen3-VL's INTERLEAVED M-RoPE -- the same three position axes as
`mrope_sectioned.py`, laid out at stride 3 instead of in contiguous blocks.

    sectioned   [T T T T | H H H H H H | W W W W W W]     Qwen2-VL
    interleaved [T H W T H W T H W T H W T H W | T]       Qwen3-VL

    axis(k) = k % 3   for k < 3 * min(section)
    axis(k) = 0       for the tail the interleave does not reach

THIS FILE EXISTS TO BE THE SAME ARITHMETIC AT A DIFFERENT ADDRESS, AND THAT IS
THE TEST. Read `mrope_sectioned.py` first -- the frequencies, the position
triple, the rotate_half pairing and the reference are all identical here. The
ONLY difference is which channel belongs to which position axis, which turns a
comparison against two boundaries into a modulo. Everything a backend does with
the numbers is unchanged; everything it does with the layout is not. Two files
rather than one because a lowering that pattern-matched the sectioned form as a
concatenation of three affine slices -- three contiguous strips is exactly the
shape a slice-fusion pass wants -- has nothing to match here, and the pair
catches that where either alone would not.

WHY QWEN3-VL MOVED IT. Sectioned M-RoPE gives time only the LOWEST frequency
bands and the spatial axes the rest. Low frequencies are the long-wavelength
ones, so temporal position is encoded almost entirely in components that barely
change between adjacent frames, and the model's sense of time degrades over a
long video exactly where it is needed. Interleaving spreads all three axes
across the WHOLE frequency range, so each one gets both the coarse and the fine
bands. The section widths survive as a budget -- how MANY channels each axis
gets -- while ceasing to decide which part of the spectrum it gets.

  The tail is real and not a rounding convenience. transformers fills the whole
  half-dim with the temporal frequencies and then overwrites `slice(1, 3*sec_h,
  3)` and `slice(2, 3*sec_w, 3)`, so when `3 * min(section)` falls short of the
  half-dim the leftover top channels stay TEMPORAL. Qwen3-VL's shipped
  `[24, 20, 20]` over 64 half-channels leaves four such channels; this kernel
  keeps one, in the same proportion, because a kernel that quietly assumed
  `HALF % 3 == 0` would be a different function.

WHAT IT ASKS OF THIS MACHINE. `k % 3` on the in-lane axis with 3 not a power of
two -- a real remainder, not a mask, unlike `sliding_window.py`'s `k & (W-1)`
or `swin_window.py`'s `& (H-1)`. It feeds a select, not an address, so it does
not become a gather; it is index arithmetic that has to survive as arithmetic.

torch equivalent: mrope_sectioned's, with `sel = pos[where(k < L, k % 3, 0), :]`.

Qwen3-VL M-RoPE: mrope_sectioned.py's arithmetic at stride 3. Same numbers, different layout -- the pair is the test, and a lowering that matched the sectioned form as three slices has nothing here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
import math

T = 32                 # tokens
D = 32                 # head dim
HALF = D // 2          # rotate_half pairs k with k + HALF
SEC_HW = 5             # channels given to H, and to W
L = 3 * SEC_HW         # the interleaved run; channels >= L stay temporal
THETA = 10000.0
INVSTEP = (2.0 / D) * math.log(THETA)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mrope_interleaved_kernel(x_ptr, pos_ptr, o_ptr,
                                 T: tl.constexpr, D: tl.constexpr,
                                 HALF: tl.constexpr, L: tl.constexpr,
                                 INVSTEP: tl.constexpr):
        t = tl.arange(0, T)
        k = tl.arange(0, HALF)

        inv = tl.exp(-k.to(tl.float32) * INVSTEP)

        pt = tl.load(pos_ptr + t).to(tl.float32)
        ph = tl.load(pos_ptr + T + t).to(tl.float32)
        pw = tl.load(pos_ptr + 2 * T + t).to(tl.float32)

        # THE INTERLEAVE. A true remainder -- 3 is not a power of two -- and a
        # tail that the interleave never reaches, which stays temporal.
        ax = tl.where(k < L, k % 3, 0)[None, :]
        pos = tl.where(ax == 0, pt[:, None],
                       tl.where(ax == 1, ph[:, None], pw[:, None]))

        ang = pos * inv[None, :]
        c = tl.cos(ang)
        s = tl.sin(ang)

        base = t[:, None] * D
        x1 = tl.load(x_ptr + base + k[None, :])
        x2 = tl.load(x_ptr + base + (HALF + k)[None, :])

        tl.store(o_ptr + base + k[None, :], x1 * c - x2 * s)
        tl.store(o_ptr + base + (HALF + k)[None, :], x2 * c + x1 * s)

    return mrope_interleaved_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randint(0, 24, (3, T), generator=g, dtype=torch.int32)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32),
            "pos": pos.reshape(-1).contiguous()}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    pos = inputs["pos"].reshape(3, T).float()
    inv = torch.exp(-torch.arange(HALF).float() * INVSTEP)[None, :]

    k = torch.arange(HALF)
    ax = torch.where(k < L, k % 3, torch.zeros_like(k))[None, :]
    sel = torch.where(ax == 0, pos[0][:, None],
                      torch.where(ax == 1, pos[1][:, None], pos[2][:, None]))
    ang = sel * inv
    c, s = ang.cos(), ang.sin()

    x1, x2 = x[:, :HALF], x[:, HALF:]
    out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=1)
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="mrope_interleaved_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "pos_ptr": "*i32", "o_ptr": "*fp32",
               "T": "constexpr", "D": "constexpr", "HALF": "constexpr",
               "L": "constexpr", "INVSTEP": "constexpr"},
    constexprs={"T": T, "D": D, "HALF": HALF, "L": L, "INVSTEP": INVSTEP},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("pos", IN, "int32", (3 * T,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
