"""Local Response Normalisation -- AlexNet's normaliser, and a channel window.

    b[c, p] = a[c, p] / (k + (alpha/n) * sum_{j in [c-2, c+2]} a[j, p]^2) ^ beta

A SLIDING WINDOW ALONG THE CHANNEL AXIS, which is what makes it worth having
here. Every other normaliser in this suite folds a whole axis: BatchNorm folds
the batch, LayerNorm folds the features, the softmax folds a row. LRN folds a
five-wide window that MOVES with the channel, so channel c's denominator
overlaps channel c+1's in four of five terms.

WHAT IT WAS FOR. AlexNet's authors describe it as "brightness normalisation"
modelled on lateral inhibition in real neurons: a strongly firing unit damps its
neighbours in channel space, so only a few channels stay large at any position
and they compete. It bought 1.4% top-1 in 2012 and was gone by 2015 -- Batch
Normalisation does the same job better, and the VGG paper reports LRN costing
memory and time for no accuracy at all.

The constants are AlexNet's: n = 5, k = 2, alpha = 1e-4, beta = 0.75.

CLAMPED, NOT WRAPPED. The window is truncated at the channel edges rather than
padded, so the first and last two channels average over fewer terms -- and the
`alpha/n` factor is NOT rescaled for them, which is the definition as published.

WHAT IT ASKS OF THIS MACHINE. With the channels on the lanes each window term
lives in a different lane, and there is no lane-to-lane move -- so the shifted
reads have to come from DRAM, five staged tiles rather than one. With the
channels in-lane it is five shifted reads inside one bank, like conv2d's taps.
Written as five explicit shifted loads for that reason: it is the same shape as
a 1-D convolution over the channel axis, and that is a shape this backend has.

torch equivalent: F.local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2)

AlexNet: a sliding window along the CHANNEL axis -- the one normaliser here that folds a moving window rather than a whole axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, HW = 32, 64
N = 5
ALPHA, BETA, KAPPA = 1e-4, 0.75, 2.0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def lrn_kernel(x_ptr, o_ptr, C: tl.constexpr, HW: tl.constexpr,
                   N: tl.constexpr, ALPHA: tl.constexpr, BETA: tl.constexpr,
                   KAPPA: tl.constexpr):
        c = tl.arange(0, C)
        p = tl.arange(0, HW)
        x = tl.load(x_ptr + c[:, None] * HW + p[None, :])

        # The window, as N shifted reads of the squares. `j` runs over the
        # offsets -2..2; the edges clamp by masking, not by wrapping.
        acc = tl.zeros((C, HW), dtype=tl.float32)
        for i in tl.static_range(N):
            j = c + (i - N // 2)
            ok = (j >= 0) & (j < C)
            jc = tl.minimum(tl.maximum(j, 0), C - 1)
            v = tl.load(x_ptr + jc[:, None] * HW + p[None, :])
            acc += tl.where(ok[:, None], v * v, 0.0)

        d = KAPPA + (ALPHA / N) * acc
        # No `pow`: beta is 0.75, so d^-0.75 = exp(-0.75 * log d) and both are
        # ops the vector unit has (see activations.py -- libdevice would become
        # a scalar libm call, which has one register per lane).
        out = x * tl.exp(-BETA * tl.log(d))
        tl.store(o_ptr + c[:, None] * HW + p[None, :], out)

    return lrn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * HW, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(1, C, HW)
    o = torch.nn.functional.local_response_norm(x, size=N, alpha=ALPHA,
                                                beta=BETA, k=KAPPA)
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="lrn_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "C": "constexpr",
               "HW": "constexpr", "N": "constexpr", "ALPHA": "constexpr",
               "BETA": "constexpr", "KAPPA": "constexpr"},
    constexprs={"C": C, "HW": HW, "N": N, "ALPHA": ALPHA, "BETA": BETA,
                "KAPPA": KAPPA},
    args=[Arg("x", IN, "float32", (C * HW,)),
          Arg("o", OUT, "float32", (C * HW,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
