"""LLaMA 3.1's RoPE scaling -- a piecewise interpolation of the frequencies.

    wavelen = 2*pi / freq
    freq' =  freq / factor            wavelen > low_freq_wavelen   (interpolate)
             freq                     wavelen < high_freq_wavelen  (leave alone)
             blend of the two         in between

EXTENDING CONTEXT IS A CHOICE BETWEEN TWO BAD OPTIONS, and this is the one that
takes neither. A model trained at 8k and run at 128k can either

    extrapolate    keep the frequencies -- the low ones now see angles the model
                   never trained on, and attention falls apart
    interpolate    divide every frequency by 16 -- positions that used to be
                   distinguishable now round together, and the HIGH frequencies,
                   which resolve neighbouring tokens, lose exactly the thing
                   they were for

The observation behind the fix: the two failures live in different frequency
bands. A dimension whose wavelength is shorter than the training context has
seen every angle it will ever see, so it needs nothing. A dimension whose
wavelength exceeds the training context has only ever seen part of one turn, so
it must be slowed down. LLaMA 3.1 interpolates the second group, leaves the
first alone, and blends across the boundary so the head dimension stays smooth.

    smooth = (ctx/wavelen - low) / (high - low)
    freq'  = (1 - smooth) * freq/factor + smooth * freq

  YaRN (Qwen2.5, and others) is the same idea with a different blend and an
  extra temperature on the logits. The shape here is the shape there.

WHAT IT ASKS OF THIS MACHINE: nothing new, and that is worth recording. It is a
piecewise select over a rank-1 frequency vector -- two comparisons and a lerp,
all inside one lane. The interesting part is that a real context extension is
elementwise arithmetic on 64 numbers, computed once per model rather than per
token, and the reason it took a year to find is that it is a statement about
which BAND matters rather than about the operation.

torch equivalent: transformers' `_compute_llama3_parameters`, then the ordinary
rotate_half.

LLaMA 3.1: context extension as a piecewise interpolation of the frequencies -- slow the long wavelengths, leave the short ones.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, D = 64, 32
H = D // 2
FACTOR = 8.0            # 8k -> 64k
LOW, HIGH = 1.0, 4.0    # low_freq_factor, high_freq_factor
OLD_CTX = 512.0         # original_max_position_embeddings, scaled down


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rope_scaled_kernel(x_ptr, o_ptr, T: tl.constexpr, D: tl.constexpr,
                           H: tl.constexpr, FACTOR: tl.constexpr,
                           LOW: tl.constexpr, HIGH: tl.constexpr,
                           OLD_CTX: tl.constexpr):
        t = tl.arange(0, T)
        h = tl.arange(0, H)

        # The base frequencies, 10000^(-2i/D).
        freq = tl.exp(-h.to(tl.float32) * (2.0 / H) * 9.210340371976184)

        # THE PIECEWISE PART. `ctx/wavelen` is how many full turns a dimension
        # makes over the training context: below LOW it has seen less than one
        # band's worth and is slowed down, above HIGH it has seen plenty.
        turns = OLD_CTX * freq / 6.283185307179586      # ctx / wavelength
        slow = freq / FACTOR
        smooth = (turns - LOW) / (HIGH - LOW)
        smooth = tl.minimum(tl.maximum(smooth, 0.0), 1.0)
        blend = (1.0 - smooth) * slow + smooth * freq
        # Outside the band the blend already equals the right end, so the
        # clamped lerp IS the three-way select -- no branch needed.
        freq = blend

        ang = t[:, None].to(tl.float32) * freq[None, :]
        cs = tl.cos(ang)
        sn = tl.sin(ang)

        lo = t[:, None] * D + h[None, :]
        hi = lo + H
        x1 = tl.load(x_ptr + lo)
        x2 = tl.load(x_ptr + hi)
        tl.store(o_ptr + lo, x1 * cs - x2 * sn)
        tl.store(o_ptr + hi, x2 * cs + x1 * sn)

    return rope_scaled_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(T * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(T, D)
    h = torch.arange(H).float()
    freq = torch.exp(-h * (2.0 / H) * 9.210340371976184)
    turns = OLD_CTX * freq / 6.283185307179586
    smooth = ((turns - LOW) / (HIGH - LOW)).clamp(0.0, 1.0)
    freq = (1 - smooth) * (freq / FACTOR) + smooth * freq
    ang = torch.arange(T).float()[:, None] * freq[None, :]
    cs, sn = ang.cos(), ang.sin()
    x1, x2 = x[:, :H], x[:, H:]
    out = torch.cat([x1 * cs - x2 * sn, x2 * cs + x1 * sn], dim=1)
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="rope_scaled_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "T": "constexpr",
               "D": "constexpr", "H": "constexpr", "FACTOR": "constexpr",
               "LOW": "constexpr", "HIGH": "constexpr",
               "OLD_CTX": "constexpr"},
    constexprs={"T": T, "D": D, "H": H, "FACTOR": FACTOR, "LOW": LOW,
                "HIGH": HIGH, "OLD_CTX": OLD_CTX},
    args=[Arg("x", IN, "float32", (T * D,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
