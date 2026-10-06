"""The SD3.5 / Flux sampler step -- a Euler step in VELOCITY space over a
resolution-shifted sigma schedule.

    sigma[i] = shift * u / (1 + (shift - 1) * u),   u = 1 - i / S     the schedule
    x'       = x + (sigma[i+1] - sigma[i]) * v                        the step

FLOW MATCHING IS NOT DIFFUSION, AND THE UPDATE RULE IS THE DIFFERENCE.
`sampler_cfg.py` is a DDIM step: the network predicts the NOISE, the schedule is
an alpha/beta variance-preserving pair, and the step goes through a predicted
clean image --

    x0 = (x - sqrt(1-a) e) / sqrt(a);   x' = sqrt(a') x0 + sqrt(1-a') e

-- three multiplies, a divide, and one scalar `t` chosen out of a table. Here the
forward process is a STRAIGHT LINE between data and noise, `x_t = (1-t) x_0 +
t eps`, so its time derivative is a constant `eps - x_0` and the network is
trained to predict that VELOCITY directly. The reverse step is then the plainest
thing in numerical analysis: `x += dt * v`, an explicit Euler step on an ODE.
Lipman et al. (2023) and Esser et al. (SD3, 2024); Flux, SD3.5 and Wan all sample
this way, and it is why they need 20-30 steps where a variance-preserving model
needed 50.

  The two are not the same arithmetic rearranged. DDIM's step is a rotation in
  (x0, eps) space with two square roots per step and no notion of dt; this is a
  difference of two schedule points times a prediction, with no square roots at
  all. A backend that got sampler_cfg.py right tells you nothing about this one.

WHAT IS ACTUALLY HARD HERE IS THE SCHEDULE, NOT THE STEP.

  1. IT IS BUILT IN-KERNEL, NOT LOOKED UP. The Euler update itself is two loads,
     a subtract, an fma -- less than sampler_cfg. The schedule is where the work
     is: S+1 sigmas from a closed form, computed as a rank-1 tile and stored.
     Every one of them is a divide by a value that depends on its own index, so
     it is a rank-1 elementwise chain with a reciprocal, which is `activations`
     shaped but feeding an ADDRESS rather than an output.

  2. THE SHIFT IS A RUNTIME SCALAR, and this is the reason the schedule cannot be
     a constexpr table. Flux computes `mu` from the image sequence length and
     sets `shift = exp(mu)`, so 512x512 and 1024x1024 get DIFFERENT sigmas from
     the same weights -- higher resolutions need more of the schedule spent at
     high noise because there is more low-frequency content to settle. Baking it
     in would recompile the sampler per resolution, the same trap
     `groupnorm_film.py`'s timestep avoids.

     The shift is a Mobius map on [0,1]: it fixes 0 and 1 and bends everything in
     between towards 1, monotonically, which is exactly the property a schedule
     reparameterisation needs.

  3. THE STEP INDEX ADDRESSES THE TILE THE KERNEL JUST WROTE. `i` arrives as an
     i32 in memory, so `sigma[i]` and `sigma[i+1]` are two scalar loads at a
     RUNTIME address out of the buffer stored three lines earlier -- there is no
     way to index a value, so the schedule has to round-trip. Two adjacent reads
     at an unknown offset, ordered after a full-tile store to the same buffer:
     the write-then-read-back dependency `moe_combine.py` has, but with a scalar
     reader instead of a tiled one, so nothing about the store's layout helps.

     It also means `dt = sigma[i+1] - sigma[i]` is NEGATIVE -- the schedule
     descends from `shift/(shift...)` at i=0 down to exactly 0 at i=S, because
     sampling runs from noise to data. A sign error here does not crash, it
     diverges.

S = 15 SO THE TABLE IS 16 LONG. `tl.arange` wants a power of two and the table
has S+1 entries, so the step count is one less than the round number. Flux ships
28 steps by default, SD3.5 ships 28 or 40.

NOT MODELLED: classifier-free guidance on `v` before the step, which is
`sampler_cfg.py` unchanged -- CFG is orthogonal to the update rule and combining
them here would just re-cover it.

A velocity-space Euler step, which is a different update rule from sampler_cfg.py's DDIM and not a rearrangement of it. The schedule is built in-kernel from a runtime shift, stored, and then read back at a runtime index -- so `sig` isolates the closed form and `o` isolates the round-trip and the sign of dt.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256                 # the latent, flattened
S = 15                  # sampler steps -- S+1 = 16 sigmas, a power of two


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def flow_match_euler_step_kernel(x_ptr, v_ptr, p_ptr, step_ptr,
                                     sig_ptr, o_ptr,
                                     N: tl.constexpr, S: tl.constexpr):
        # THE SCHEDULE, built here rather than looked up, because `shift`
        # depends on the resolution and would otherwise force a recompile.
        ri = tl.arange(0, S + 1)
        shift = tl.load(p_ptr)
        u = 1.0 - ri.to(tl.float32) / S
        sig = shift * u / (1.0 + (shift - 1.0) * u)
        tl.store(sig_ptr + ri, sig)

        # THE STEP INDEX, out of memory, addressing the tile just stored: a
        # value cannot be indexed, so the schedule round-trips through DRAM and
        # comes back as two scalars at a runtime offset.
        i = tl.load(step_ptr)
        s0 = tl.load(sig_ptr + i)
        s1 = tl.load(sig_ptr + i + 1)

        # EULER IN VELOCITY SPACE. dt is negative: sigma descends to zero.
        j = tl.arange(0, N)
        tl.store(o_ptr + j,
                 tl.load(x_ptr + j) + (s1 - s0) * tl.load(v_ptr + j))

    return flow_match_euler_step_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "v": torch.randn(N, generator=g, dtype=torch.float32),
            # shift = exp(mu); 3.0 is what Flux resolves to around 1024x1024.
            "p": torch.tensor([3.0], dtype=torch.float32),
            # mid-schedule on purpose: neither endpoint, so an off-by-one in
            # the sigma indexing changes the answer.
            "step": torch.tensor([6], dtype=torch.int32)}


def reference(inputs):
    import torch
    shift = inputs["p"][0]
    u = 1.0 - torch.arange(S + 1, dtype=torch.float32) / S
    sig = shift * u / (1.0 + (shift - 1.0) * u)
    i = int(inputs["step"][0])
    return {"sig": sig.contiguous(),
            "o": (inputs["x"] + (sig[i + 1] - sig[i]) * inputs["v"]
                  ).contiguous()}


SPEC = KernelSpec(
    name="flow_match_euler_step_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "v_ptr": "*fp32", "p_ptr": "*fp32",
               "step_ptr": "*i32", "sig_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr", "S": "constexpr"},
    constexprs={"N": N, "S": S},
    args=[Arg("x", IN, "float32", (N,)),
          Arg("v", IN, "float32", (N,)),
          Arg("p", IN, "float32", (1,)),
          Arg("step", IN, "int32", (1,)),
          Arg("sig", OUT, "float32", (S + 1,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
)
