"""A DDIM step with classifier-free guidance -- the loop, not the network.

    e   = e_u + w * (e_c - e_u)                       CFG: two batches combined
    x0  = (x - sqrt(1-a) * e) / sqrt(a)               the predicted clean image
    x'  = sqrt(a') * x0 + sqrt(1-a') * e              re-noised to the next step

EVERYTHING ELSE IN THESE THREE SUITES IS INSIDE A NETWORK. This is the arithmetic
BETWEEN two calls to one, and it is why diffusion is a loop rather than a model:
the same weights run 20 to 50 times and this runs between them.

CLASSIFIER-FREE GUIDANCE IS A SUBTRACTION BETWEEN TWO BATCHES. The network is
called twice -- once with the text conditioning and once with an empty prompt --
and the two predictions are extrapolated apart:

    e = e_u + w (e_c - e_u)          w = 7.5 is a common choice

  What it does: `e_c - e_u` is the direction the conditioning pushes, and w
  pushes further along it than the model actually predicted. It is why a
  guidance scale above 1 sharpens prompt adherence and above ~15 starts
  producing saturated, over-committed images. Ho & Salimans (2022) introduced it
  to remove the separate classifier earlier guidance needed.

  In practice the two are batched into ONE network call, so the batch axis is
  really "which conditioning", and this kernel is where it collapses.

THE BATCH AXIS IS WHAT THIS BACKEND HAS NOT SEEN. Every kernel in all three
lists handles one batch. Here two of them are read by the same kernel and
combined elementwise -- and because Triton cannot slice a value, they are two
LOADS at different offsets rather than one tile with a batch axis. That is also
what a real implementation does (`eps.chunk(2)`), and it keeps the tile rank 1.

  Had it been one `[2, N]` tile, the combine would be a fold over an extent-2
  axis and the answer would depend on where the batch landed: on the lanes it is
  a lane-to-lane move and a DRAM round trip, in-lane it is a subtraction. Same
  trade `convnext_block` makes with `[HW, C]`.

THE SCHEDULE COMES FROM A BUFFER, not from constexprs. `alpha_t` changes every
step, and baking it in would recompile the network once per step. A real sampler
indexes a precomputed table by `t`; this takes the four numbers that table
lookup would have produced.

torch equivalent:
    e = e_u + w * (e_c - e_u)
    x0 = (x - sm * e) / sa
    x_prev = sap * x0 + smp * e

The sampler loop rather than the network: classifier-free guidance collapsing two batches, and a schedule read from a buffer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024                # the latent, flattened
W = 7.5                 # guidance scale


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def sampler_cfg_kernel(x_ptr, e_ptr, s_ptr, o_ptr,
                           N: tl.constexpr, W: tl.constexpr):
        i = tl.arange(0, N)

        # TWO BATCHES, TWO LOADS. `eps` is [2, N] in DRAM and the two halves are
        # read at their own offsets -- a value cannot be sliced, and chunking is
        # what the reference implementation does anyway.
        eu = tl.load(e_ptr + i)
        ec = tl.load(e_ptr + N + i)
        e = eu + W * (ec - eu)

        # THE SCHEDULE, from a buffer: four numbers a table lookup by `t` would
        # have produced. sqrt(a), sqrt(1-a), sqrt(a'), sqrt(1-a').
        sa = tl.load(s_ptr + 0)
        sm = tl.load(s_ptr + 1)
        sap = tl.load(s_ptr + 2)
        smp = tl.load(s_ptr + 3)

        x = tl.load(x_ptr + i)
        x0 = (x - sm * e) / sa               # the predicted clean latent
        tl.store(o_ptr + i, sap * x0 + smp * e)

    return sampler_cfg_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    a, ap = 0.36, 0.49
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "e": torch.randn(2 * N, generator=g, dtype=torch.float32),
            "s": torch.tensor([a ** 0.5, (1 - a) ** 0.5,
                               ap ** 0.5, (1 - ap) ** 0.5],
                              dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    eu, ec = inputs["e"][:N], inputs["e"][N:]
    e = eu + W * (ec - eu)
    sa, sm, sap, smp = inputs["s"].tolist()
    x0 = (x - sm * e) / sa
    return {"o": (sap * x0 + smp * e).contiguous()}


SPEC = KernelSpec(
    name="sampler_cfg_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "e_ptr": "*fp32", "s_ptr": "*fp32",
               "o_ptr": "*fp32", "N": "constexpr", "W": "constexpr"},
    constexprs={"N": N, "W": W},
    args=[Arg("x", IN, "float32", (N,)),
          Arg("e", IN, "float32", (2 * N,)),
          Arg("s", IN, "float32", (4,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
