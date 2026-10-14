"""Whisper's log-mel: a filterbank matmul, a log, and a floor tied to the
GLOBAL maximum of the result.

whisper/audio.py, verbatim after the STFT:

    mel_spec = filters @ magnitudes                    [MEL, BINS] @ [BINS, F]
    log_spec = clamp(mel_spec, min=1e-10).log10()
    log_spec = maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

`magnitudes` is `stft.abs() ** 2` -- power, not amplitude -- so this kernel's
input is what `stft_frame.py` produces, squared, and the two together are the
entire Whisper front end. There is nothing else in front of the encoder.

THE FILTERBANK IS A MATMUL WITH A BAND STRUCTURE NOTHING EXPLOITS. Row `m` is a
triangle over three mel-spaced corner frequencies, so it is nonzero on a handful
of adjacent bins and zero everywhere else -- at Whisper's real size, 80 by 201,
about 95% of the matrix is structural zero. It is still shipped and multiplied
as a dense `[80, 201]` matrix by every implementation, because a banded matmul
at this size is slower than a dense one and the bank is a constant. This kernel
keeps that: a dense `tl.dot` whose left operand is mostly zeros, which is the
honest thing to measure.

THE FLOOR IS THE PART WITH NO ANALOGUE HERE. `log_spec.max()` is a reduction
over BOTH axes of the matmul's result, and the value it produces is then
subtracted from every element. So:

  * it is a GLOBAL reduction, and this machine has no lane-crossing fold. It is
    written as two INNER-axis reductions -- `tl.max(axis=1)` down to one value
    per mel band, then a second `tl.max(axis=1)` over that rank-1 tensor viewed
    as a `[1, MEL]` tile, which is the only way to fold along what was the lane
    axis. The rank-1 intermediate CHANGES WHICH AXIS IT LIVES ON between the two
    steps, and that transpose is the whole cost of a global max here.
  * it makes the kernel untileable. `reduce_row_max.py` folds a row and every
    row is independent; here element (0, 0) depends on element (15, 31), so a
    program that owned half the frames would compute a different floor. Whisper
    takes the max over the whole 30-second window, which is why the mel of a
    clip is not the concatenation of the mels of its halves.

CLAMP BEFORE LOG, FLOOR AFTER. `clamp(min=1e-10)` stops `log10(0) = -inf` from
poisoning the max that follows -- with an -inf in the tile, `max - 8` is still
finite but every clamped element would come back as -inf and the ordering of the
two operations would stop mattering. It matters: the clamp is an absolute floor
at -10 decades, the `max - 8` is a RELATIVE one at 8 decades below the loudest
thing in the utterance, and quiet audio is where they disagree.

`log10` is `tl.log(x) * (1 / ln 10)`; this triton has no base-10 log.

Nearest neighbours: `softmax.py` also pairs a max with a broadcast, but its max
is per-row and its consumer is an exp. Here the reduction crosses the tile and
the consumer is a `maximum`, so nothing normalises and the result of the fold is
a floor rather than a shift.

A matmul feeding a log feeding a floor that depends on the max of the ENTIRE result -- a two-stage global reduction, and the one kernel here that a frame-wise split would silently change. `mel` is the matmul alone, so the two failure modes separate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

MEL = 16          # mel bands (Whisper: 80, or 128 for large-v3)
BINS = 64         # spectrogram bins (Whisper: 201)
F = 32            # frames
LN10 = 2.302585092994046
SR = 16000.0      # sample rate, only used to build the filterbank


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mel_logspec_kernel(fb_ptr, mag_ptr, mel_ptr, o_ptr,
                           MEL: tl.constexpr, BINS: tl.constexpr,
                           F: tl.constexpr, LN10: tl.constexpr):
        m = tl.arange(0, MEL)
        b = tl.arange(0, BINS)
        t = tl.arange(0, F)

        fb = tl.load(fb_ptr + m[:, None] * BINS + b[None, :])     # [MEL, BINS]
        mag = tl.load(mag_ptr + b[:, None] * F + t[None, :])      # [BINS, F]
        mel = tl.dot(fb, mag)                                     # [MEL, F]
        tl.store(mel_ptr + m[:, None] * F + t[None, :], mel)

        # log10 of the clamped power.
        ls = tl.log(tl.maximum(mel, 1e-10)) / LN10

        # GLOBAL MAX IN TWO INNER-AXIS FOLDS. The first is per mel band; the
        # second reduces the rank-1 result, which has to be viewed as a [1, MEL]
        # tile to put the surviving axis in-lane -- there is no fold across the
        # lanes.
        band = tl.max(ls, axis=1)                                 # [MEL]
        gmax = tl.max(band[None, :], axis=1)                      # [1]

        # 8 decades below the loudest point of the WHOLE utterance, then the
        # affine that puts the result roughly in [-1, 1].
        ls = tl.maximum(ls, gmax[:, None] - 8.0)
        tl.store(o_ptr + m[:, None] * F + t[None, :], (ls + 4.0) / 4.0)

    return mel_logspec_kernel


def _filterbank(torch):
    """Triangular mel bank, htk scale: mel = 2595 log10(1 + f / 700)."""
    freqs = torch.linspace(0.0, SR / 2.0, BINS, dtype=torch.float64)

    def hz2mel(f):
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel2hz(mu):
        return 700.0 * (10.0 ** (mu / 2595.0) - 1.0)

    pts = mel2hz(torch.linspace(hz2mel(torch.tensor(0.0)),
                                hz2mel(torch.tensor(SR / 2.0)),
                                MEL + 2, dtype=torch.float64))
    fdiff = pts[1:] - pts[:-1]
    ramps = pts[:, None] - freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = torch.clamp(torch.minimum(lower, upper), min=0.0)
    # Slaney area normalisation, as librosa (and therefore Whisper) uses.
    w = w * (2.0 / (pts[2:] - pts[:-2]))[:, None]
    return w.to(torch.float32).contiguous()


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # magnitudes = |STFT|^2, so nonnegative -- and shaped like real audio, which
    # is what makes the -8 floor fire. White noise here would span less than one
    # decade after the filterbank and the `maximum` would be a no-op, testing
    # nothing. Four decades of spectral tilt plus a stretch of near-silent
    # frames puts the quiet corner well below the loud one.
    mag = torch.randn(BINS, F, generator=g, dtype=torch.float32) ** 2
    tilt = 10.0 ** (-4.0 * torch.arange(BINS, dtype=torch.float32) / BINS)
    gain = torch.ones(F, dtype=torch.float32)
    gain[F // 2: F // 2 + 6] = 1e-6                    # a silent stretch
    mag = (mag * tilt[:, None] * gain[None, :]).contiguous()
    return {"fb": _filterbank(torch).reshape(-1), "mag": mag.reshape(-1)}


def reference(inputs):
    fb = inputs["fb"].reshape(MEL, BINS)
    mag = inputs["mag"].reshape(BINS, F)
    mel = fb @ mag
    ls = mel.clamp(min=1e-10).log10()
    ls = ls.maximum(ls.max() - 8.0)
    return {"mel": mel.reshape(-1).contiguous(),
            "o": ((ls + 4.0) / 4.0).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="mel_logspec_kernel", kernel=kernel,
    signature={"fb_ptr": "*fp32", "mag_ptr": "*fp32", "mel_ptr": "*fp32",
               "o_ptr": "*fp32", "MEL": "constexpr", "BINS": "constexpr",
               "F": "constexpr", "LN10": "constexpr"},
    constexprs={"MEL": MEL, "BINS": BINS, "F": F, "LN10": LN10},
    args=[Arg("fb", IN, "float32", (MEL * BINS,)),
          Arg("mag", IN, "float32", (BINS * F,)),
          Arg("mel", OUT, "float32", (MEL * F,)),
          Arg("o", OUT, "float32", (MEL * F,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    # THE GLOBAL MAX, A SCALAR, SPREAD OVER A 16x32 TILE: the unit's replicate
    # lands it once per lane and the bank fills the rest; before, through DRAM.
    expect_ir={"08-transformed.mlir": {"dram_arg = -1": 0}},
)
