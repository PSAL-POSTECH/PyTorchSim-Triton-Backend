"""The windowed DFT -- the first thing Whisper does to audio, and the first
Fourier transform in this suite.

    frame[f, n] = x[f * HOP + n]                   overlapping reads
    xw          = frame * hann(n)
    re[f, k]    = sum_n xw[f, n] * cos(2 pi k n / N)
    im[f, k]    = -sum_n xw[f, n] * sin(2 pi k n / N)
    mag         = sqrt(re^2 + im^2)

whisper/audio.py is `torch.stft(audio, N_FFT, HOP_LENGTH,
window=torch.hann_window(N_FFT), return_complex=True)` with N_FFT=400,
HOP_LENGTH=160 at 16 kHz, and every downstream number in the model is derived
from it. NOTHING IN THIS SUITE HAD AN FFT, A DFT, OR A WINDOW. Speech models
are the only family that starts from a signal rather than a token table, and the
whole front end -- this kernel, then mel_logspec.py -- was missing.

THE TRANSFORM IS A MATMUL HERE, ON PURPOSE. An FFT is O(N log N) and a DFT
matmul is O(N^2), and at N = 64 that is 64 versus about 6 multiply-adds per
output -- but the FFT's saving is bought with a log N-deep chain of butterflies
whose data movement is a bit-reversal permutation and a different stride per
stage. That is `sort_bitonic.py`'s shape, which this backend refuses. A DFT
written as a dense `[N, N]` real matrix and a second one for the imaginary part
is two `tl.dot`s that lower today, and it is what a matrix unit wants. The
cost is real and the tradeoff is the point: on this machine the naive transform
is the fast one.

  AND THE TWO TRANSFORMS ARE NOT BIT-IDENTICAL. This reference agrees with
  `torch.stft`, which is an FFT, to 7.1e-5 absolute on bins whose magnitude
  reaches 11 -- about 6e-6 relative. That gap is the summation ORDER, nothing
  else, and it is the floor on how tightly any implementation of this kernel can
  be held against a library.

  TWO MATMULS BECAUSE THERE IS NO COMPLEX DTYPE. cos and sin are separate
  tables, `re` and `im` are separate tiles, and the only place they meet is the
  magnitude. A backend with a complex type would fuse them; here the
  interleaving is written out, and the sign convention (`e^{-i.}`, so `im` is
  NEGATED) is the one thing that is easy to get wrong and invisible in `mag`.

FRAMING IS AN OVERLAPPING READ. Row `f` starts at `f * HOP` with HOP < N, so
consecutive rows of the loaded tile share half their samples: one address
expression, `f * HOP + n`, whose rows ALIAS. It is still perfectly affine -- a
2-D read with row stride HOP and element stride 1 out of a 1-D buffer -- so it
costs nothing beyond reading the signal twice. `istft_overlap_add.py` is the
same map on the WRITE side, where aliasing rows are no longer free.

33 BINS, 64 WIDE. The real DFT of 64 samples has 33 non-redundant bins
(`N // 2 + 1`); bins 33..63 are conjugate mirrors of 31..1 and carry nothing.
33 is not a power of two and cannot be a tile axis, so the bin axis is the full
64, the mirror half is computed and thrown away, and the store carries
`k < BINS`. Half the transform's work is discarded -- which is exactly what
`torch.fft.rfft` avoids and a dense matmul cannot.

The Hann window is built from `tl.cos` of an index, not loaded: periodic Hann is
`0.5 - 0.5 cos(2 pi n / N)`, which is what `torch.hann_window(N)` returns
(periodic=True is the default, and it is the right one for STFT -- the symmetric
variant would break the overlap-add on the way back). The twiddle tables ARE
loaded, because they are what makes this an honest matmul rather than a
transcendental per output element.

The suite's first DFT: an overlapping framing read, a windowing built from indices, two real matmuls standing in for one complex one, and a masked store that drops the conjugate half. `re` and `im` are linear in the input, so a mismatch there is the transform; a mismatch only in `mag` is the sqrt.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NFRAME = 16                      # frames in one tile
N = 64                           # n_fft (Whisper: 400)
HOP = 32                         # hop_length, N // 2 (Whisper: 160)
BINS = N // 2 + 1                # 33 -- the non-redundant half
NSAMP = (NFRAME - 1) * HOP + N   # 544 samples of signal


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def stft_frame_kernel(x_ptr, cw_ptr, sw_ptr, re_ptr, im_ptr, mag_ptr,
                          NFRAME: tl.constexpr, N: tl.constexpr,
                          HOP: tl.constexpr, BINS: tl.constexpr):
        f = tl.arange(0, NFRAME)[:, None]
        n = tl.arange(0, N)[None, :]

        # FRAMING: row stride HOP < N, so consecutive rows overlap by N - HOP
        # samples. Affine, and the aliasing is only a re-read.
        frame = tl.load(x_ptr + f * HOP + n)

        # Periodic Hann, from the index -- no table.
        w = 0.5 - 0.5 * tl.cos(6.283185307179586 * n.to(tl.float32) / N)
        xw = frame * w

        nn = tl.arange(0, N)[:, None]
        kk = tl.arange(0, N)[None, :]
        cw = tl.load(cw_ptr + nn * N + kk)          # cos(2 pi k n / N)
        sw = tl.load(sw_ptr + nn * N + kk)          # sin(2 pi k n / N)

        # X[k] = sum_n x[n] (cos - i sin): the minus sign lives here.
        re = tl.dot(xw, cw)
        im = -tl.dot(xw, sw)
        mag = tl.sqrt(re * re + im * im)

        # Keep 33 of the 64 bins. The other 31 are conjugate mirrors.
        k = tl.arange(0, N)[None, :]
        off = f * BINS + k
        keep = tl.broadcast_to(k < BINS, (NFRAME, N))
        tl.store(re_ptr + off, re, mask=keep)
        tl.store(im_ptr + off, im, mask=keep)
        tl.store(mag_ptr + off, mag, mask=keep)

    return stft_frame_kernel


def _twiddle(torch):
    n = torch.arange(N, dtype=torch.float32)[:, None]
    k = torch.arange(N, dtype=torch.float32)[None, :]
    ang = 2.0 * 3.141592653589793 * k * n / N
    return ang.cos().contiguous(), ang.sin().contiguous()


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    cw, sw = _twiddle(torch)
    return {"x": torch.randn(NSAMP, generator=g, dtype=torch.float32),
            "cw": cw.reshape(-1), "sw": sw.reshape(-1)}


def reference(inputs):
    import torch
    x = inputs["x"]
    f = torch.arange(NFRAME)[:, None]
    n = torch.arange(N)[None, :]
    frame = x[f * HOP + n]                                  # [NFRAME, N]
    # torch.hann_window(N) -- periodic, i.e. 0.5 - 0.5 cos(2 pi n / N)
    w = 0.5 - 0.5 * torch.cos(2.0 * 3.141592653589793
                              * torch.arange(N, dtype=torch.float32) / N)
    xw = frame * w[None, :]
    cw = inputs["cw"].reshape(N, N)
    sw = inputs["sw"].reshape(N, N)
    re = xw @ cw
    im = -(xw @ sw)
    mag = (re * re + im * im).sqrt()
    return {"re": re[:, :BINS].reshape(-1).contiguous(),
            "im": im[:, :BINS].reshape(-1).contiguous(),
            "mag": mag[:, :BINS].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="stft_frame_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "cw_ptr": "*fp32", "sw_ptr": "*fp32",
               "re_ptr": "*fp32", "im_ptr": "*fp32", "mag_ptr": "*fp32",
               "NFRAME": "constexpr", "N": "constexpr", "HOP": "constexpr",
               "BINS": "constexpr"},
    constexprs={"NFRAME": NFRAME, "N": N, "HOP": HOP, "BINS": BINS},
    args=[Arg("x", IN, "float32", (NSAMP,)),
          Arg("cw", IN, "float32", (N * N,)),
          Arg("sw", IN, "float32", (N * N,)),
          Arg("re", OUT, "float32", (NFRAME * BINS,)),
          Arg("im", OUT, "float32", (NFRAME * BINS,)),
          Arg("mag", OUT, "float32", (NFRAME * BINS,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
