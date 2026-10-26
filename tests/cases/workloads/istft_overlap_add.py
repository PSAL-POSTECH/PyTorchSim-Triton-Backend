"""The inverse STFT a neural vocoder ends with: polar to rectangular, an inverse
DFT matmul, and an OVERLAP-ADD whose writes collide.

    re[f, k] = mag[f, k] * cos(phase[f, k])
    im[f, k] = mag[f, k] * sin(phase[f, k])
    fr[f, n] = (sum_k re * cos(2 pi k n / N) - im * sin(2 pi k n / N)) / N
    fr      *= hann(n)                                     synthesis window
    y[f * HOP + n] += fr[f, n]                             OVERLAP-ADD

WHO RUNS THIS. Vocos, iSTFTNet and HiFi-GAN's iSTFT head all stop the network at
a magnitude and a phase per bin per frame and hand the last step to a fixed
inverse transform -- no learned upsampling, no transposed convolutions. The
argument is that a vocoder's hard part is predicting phase, not synthesising
samples, so once the phase exists the waveform is arithmetic. That makes this
kernel the LAST thing a TTS or speech-enhancement stack runs, and the only one
whose output is audio.

POLAR TO RECTANGULAR IS WHY THE NETWORK'S OUTPUT IS TWO TENSORS. A magnitude is
positive and a phase wraps, so both are easy to predict and neither is a
coordinate the transform can use. `mag * cos(phase)` and `mag * sin(phase)` are
two transcendentals per bin per frame, done on the FULL tile before any
reduction -- the widest transcendental work in this suite, and the point where a
backend that vectorises `exp` but scalarises `sin` shows it.

THE OVERLAP-ADD IS THE KERNEL. Frame `f` lands at `f * HOP` and is `N` long with
HOP < N, so frames `f` and `f+1` share `N - HOP` output samples and BOTH write
them. `stft_frame.py` has the same address map on the read side, where aliasing
rows are just a re-read; here they are a data race, and the difference between
the two files is entirely which side of the memory the aliasing is on.

  WRITTEN AS TWO INTERLEAVED PASSES, NOT AS AN ATOMIC. With HOP = N/2 every
  output sample is touched by exactly two frames, and they have opposite parity.
  So the even frames are disjoint from each other, the odd frames are disjoint
  from each other, and each pass is an ordinary strided store: row `j` of pass
  `p` writes `j * 2HOP + p * HOP + n`. The collision survives only ACROSS the
  two passes, where the static loop already orders it. That is exactly the trick
  `moe_combine.py` documents for the MoE scatter -- group the writes so no tile
  collides with itself -- and it is why neither kernel needs an atomic. A
  general HOP (Whisper's 160 into 400, so between two and three frames deep)
  needs ceil(N/HOP) passes by the same argument.

  THE FRAMES GO THROUGH MEMORY TO GET THERE. The inverse transform is one
  `[NFRAME, N]` matmul -- splitting it by parity would halve the rows and take
  `tl.dot` below its minimum -- but the overlap-add wants one PARITY of the rows
  at a time, and a register tile has no dynamic row index. So `fr` is stored and
  read back with a row stride of `2N`. It is an output anyway: a wrong `fr` is
  the transform and a wrong `y` with a right `fr` is the add, and that is the
  only way to tell them apart from the audio.

  THE ZEROING IS NOT CEREMONY. Both passes accumulate, so the buffer starts at
  zero, and `(NFRAME-1)*HOP + N` is not a power of two -- it is zeroed through
  the same two-pass address map, whose union is exactly the written range.

THE SPECTRUM IS HERMITIAN, AND NOTHING HERE USES IT. A real waveform's DFT
satisfies `X[N-k] = conj(X[k])`, so a vocoder predicts `N/2 + 1` bins and the
mirror is implied; `make_inputs` builds the mirror explicitly and the kernel
transforms all N bins, taking the real part. Exploiting the symmetry would halve
the matmul and is what `irfft` does -- and it would also hide the sign
convention, which is the one thing an inverse transform gets wrong silently.

NOT MODELLED: the `1 / sum(w^2)` envelope division `torch.istft` applies after
the add. It is a per-sample multiply by a host-computable constant vector and it
moves no data; leaving it out keeps the reconstruction error attributable.

CURRENTLY FAILING, AND THE WRONG NUMBER MOVES WITH THE SCRATCHPAD LAYOUT
This kernel does not verify, and the value it comes back with depends on where
the scratchpad globals SIT, which means something here reads or writes outside
the tile it owns. Measured by changing nothing but the reservation -- the reservation
stopped guessing each buffer's lane spread from the narrowest dim and started
reading `pytorchsim.lane_axis`, which shrank this kernel's six [16, 64] axis-1 tiles
from 64 to 16 elements per lane (64 is the wrong number: 1024 elements over 64
lanes is 16 each):

    .spad section    3360 -> 2208 bytes/lane
    max_abs_err      0.180213 0.181941  ->  0.188028 0.1938

with 08-transformed.mlir and 10-lowered.mlir BYTE-IDENTICAL between the two apart
from the symbol names. Identical IR and a different answer can only be storage:
the old over-reservation was padding an overrun, not preventing one, and 41
other kernels shrank the same way with every number unchanged. Deterministic --
the same two values on repeated sweeps.

NOT ISOLATED. The overrun was not tracked to an op; this is the smallest form it
has been seen in, and it is recorded rather than reduced (rule 2). What would
find it: dump the `.spad` section after a run and look for a write past a tile's
16-element share, most likely on one of the buf2..buf9 axis-1 tiles above.

Overlapping WRITES: two parity passes into one waveform, the collision living only between them. stft_frame.py is the same address map on the read side. `fr` separates the inverse transform from the add.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NFRAME = 16                      # frames
N = 64                           # n_fft
HOP = 32                         # N // 2 -- two frames per output sample
NSAMP = (NFRAME - 1) * HOP + N   # 544
HALF = NFRAME // 2               # rows per parity pass


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def istft_overlap_add_kernel(mag_ptr, ph_ptr, ic_ptr, isn_ptr,
                                 fr_ptr, y_ptr,
                                 NFRAME: tl.constexpr, N: tl.constexpr,
                                 HOP: tl.constexpr, HALF: tl.constexpr):
        f = tl.arange(0, NFRAME)[:, None]
        k = tl.arange(0, N)[None, :]

        # POLAR -> RECTANGULAR: two transcendentals over the whole tile.
        mag = tl.load(mag_ptr + f * N + k)
        ph = tl.load(ph_ptr + f * N + k)
        re = mag * tl.cos(ph)
        im = mag * tl.sin(ph)

        kk = tl.arange(0, N)[:, None]
        nn = tl.arange(0, N)[None, :]
        ic = tl.load(ic_ptr + kk * N + nn)          # cos(2 pi k n / N) / N
        isn = tl.load(isn_ptr + kk * N + nn)        # sin(2 pi k n / N) / N

        # Real part of the inverse transform, then the synthesis window --
        # periodic Hann again, from the index, matching stft_frame.py.
        fr = tl.dot(re, ic) - tl.dot(im, isn)
        w = 0.5 - 0.5 * tl.cos(6.283185307179586 * k.to(tl.float32) / N)
        fr = fr * w
        tl.store(fr_ptr + f * N + k, fr)

        j = tl.arange(0, HALF)[:, None]
        n = tl.arange(0, N)[None, :]

        # Zero the whole written range first: the two passes' address maps
        # union to [0, (NFRAME-1)*HOP + N), which is not a power of two.
        for p in tl.static_range(2):
            tl.store(y_ptr + j * (2 * HOP) + p * HOP + n,
                     tl.zeros((HALF, N), dtype=tl.float32))

        # OVERLAP-ADD, one parity at a time. Inside a pass the rows are 2*HOP
        # apart and cannot collide; between the passes they always do, and the
        # loop is what orders them.
        for p in tl.static_range(2):
            v = tl.load(fr_ptr + (2 * j + p) * N + n)
            off = j * (2 * HOP) + p * HOP + n
            tl.store(y_ptr + off, tl.load(y_ptr + off) + v)

    return istft_overlap_add_kernel


def _itwiddle(torch):
    k = torch.arange(N, dtype=torch.float32)[:, None]
    n = torch.arange(N, dtype=torch.float32)[None, :]
    ang = 2.0 * 3.141592653589793 * k * n / N
    return (ang.cos() / N).contiguous(), (ang.sin() / N).contiguous()


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # A real signal's spectrum: |X[N-k]| = |X[k]| and arg X[N-k] = -arg X[k],
    # with bins 0 and N/2 real. This is the N/2+1 bins a vocoder predicts,
    # mirrored out to the full N the kernel transforms.
    mag = torch.rand(NFRAME, N // 2 + 1, generator=g, dtype=torch.float32)
    ph = (torch.rand(NFRAME, N // 2 + 1, generator=g, dtype=torch.float32)
          * 6.283185307179586 - 3.141592653589793)
    ph[:, 0] = 0.0
    ph[:, N // 2] = 0.0
    full_mag = torch.cat([mag, mag[:, 1:N // 2].flip(1)], dim=1)
    full_ph = torch.cat([ph, -ph[:, 1:N // 2].flip(1)], dim=1)
    ic, isn = _itwiddle(torch)
    return {"mag": full_mag.contiguous().reshape(-1),
            "ph": full_ph.contiguous().reshape(-1),
            "ic": ic.reshape(-1), "isn": isn.reshape(-1)}


def reference(inputs):
    import torch
    mag = inputs["mag"].reshape(NFRAME, N)
    ph = inputs["ph"].reshape(NFRAME, N)
    re = mag * ph.cos()
    im = mag * ph.sin()
    ic = inputs["ic"].reshape(N, N)
    isn = inputs["isn"].reshape(N, N)
    n = torch.arange(N, dtype=torch.float32)
    w = 0.5 - 0.5 * torch.cos(2.0 * 3.141592653589793 * n / N)
    fr = (re @ ic - im @ isn) * w[None, :]
    y = torch.zeros(NSAMP, dtype=torch.float32)
    for f in range(NFRAME):
        y[f * HOP: f * HOP + N] += fr[f]
    return {"fr": fr.reshape(-1).contiguous(), "y": y}


SPEC = KernelSpec(
    name="istft_overlap_add_kernel", kernel=kernel,
    signature={"mag_ptr": "*fp32", "ph_ptr": "*fp32", "ic_ptr": "*fp32",
               "isn_ptr": "*fp32", "fr_ptr": "*fp32", "y_ptr": "*fp32",
               "NFRAME": "constexpr", "N": "constexpr", "HOP": "constexpr",
               "HALF": "constexpr"},
    constexprs={"NFRAME": NFRAME, "N": N, "HOP": HOP, "HALF": HALF},
    args=[Arg("mag", IN, "float32", (NFRAME * N,)),
          Arg("ph", IN, "float32", (NFRAME * N,)),
          Arg("ic", IN, "float32", (N * N,)),
          Arg("isn", IN, "float32", (N * N,)),
          Arg("fr", OUT, "float32", (NFRAME * N,)),
          Arg("y", OUT, "float32", (NSAMP,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
