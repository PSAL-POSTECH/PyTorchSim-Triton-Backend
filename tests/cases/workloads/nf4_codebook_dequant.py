"""bitsandbytes NF4 with double quantisation -- dequant is a TABLE LOOKUP, twice."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

K, N, G = 32, 16, 8          # in features, out features, weights per absmax
NB = K // G                  # absmax blocks per column

# bitsandbytes' NF4 codebook: the 16 quantiles of a standard normal, rescaled so
# the extremes are exactly -1 and +1. Index 7 is exactly zero on purpose.
NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
       -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
       0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
       0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236023, 1.0)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def nf4_codebook_dequant_kernel(wq_ptr, aq_ptr, c1_ptr, c2_ptr, s2_ptr,
                                    off_ptr, am_ptr, w_ptr, K: tl.constexpr,
                                    N: tl.constexpr, G: tl.constexpr,
                                    NB: tl.constexpr):
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        rb = tl.arange(0, NB)

        s2 = tl.load(s2_ptr + rn)
        off = tl.load(off_ptr + rn)

        # PASS 2 -- the absmax array is itself quantised. A [NB, N] tile of
        # 4-bit indices, gathered through the second-level table, then rescaled
        # affinely. Stored so the intermediate is checkable on its own.
        aq_b = tl.load(aq_ptr + rb[:, None] * N + rn[None, :])        # [NB, N]
        am_b = tl.load(c2_ptr + aq_b) * s2[None, :] + off[None, :]
        tl.store(am_ptr + rb[:, None] * N + rn[None, :], am_b)

        # The same lookup again at the granularity pass 1 needs. THE COARSE-AXIS
        # STRETCH IS ON THE INDEX, not on the dequantised value: `rk // G` picks
        # the block, and the gather is then issued at [K, N]. Doing it the other
        # way would need a rank-3 broadcast of am_b, and it would not be what a
        # real kernel does -- the index is a byte, the value is a float.
        aq_k = tl.load(aq_ptr + (rk[:, None] // G) * N + rn[None, :])  # [K, N]
        am_k = tl.load(c2_ptr + aq_k) * s2[None, :] + off[None, :]

        # PASS 1 -- the weights. Two nibbles per container, and the index is
        # UNSIGNED: unlike an affine int4 there is no bias by 8, because the
        # sign is a property of the table entry and not of the bit pattern.
        byte = tl.load(wq_ptr + (rk[:, None] // 2) * N + rn[None, :])
        idx = (byte >> ((rk[:, None] % 2) * 4)) & 0xF                 # [K, N]

        # A 512-element gather into a 16-entry table. Not an arithmetic
        # reconstruction of the value -- there is no formula for these levels.
        # Its scale is the OTHER gather's result: the chain is data, not layout.
        lvl = tl.load(c1_ptr + idx)                                   # [K, N]
        tl.store(w_ptr + rk[:, None] * N + rn[None, :], lvl * am_k)

    return nf4_codebook_dequant_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # The second-level table: positive, unequally spaced, no closed form --
    # standing in for bitsandbytes' dynamic-exponent absmax code.
    c2 = torch.tensor([0.02 * (1.35 ** i) for i in range(16)],
                      dtype=torch.float32)
    return {"wq": torch.randint(0, 256, (K // 2, N), generator=g,
                                dtype=torch.int32),
            "aq": torch.randint(0, 16, (NB, N), generator=g, dtype=torch.int32),
            "c1": torch.tensor(NF4, dtype=torch.float32),
            "c2": c2,
            "s2": torch.rand(N, generator=g, dtype=torch.float32) * 0.5 + 0.25,
            "off": torch.rand(N, generator=g, dtype=torch.float32) * 0.05}


def reference(inputs):
    import torch
    aq = inputs["aq"].to(torch.int64)
    am = inputs["c2"][aq] * inputs["s2"][None, :] + inputs["off"][None, :]

    byte = inputs["wq"]
    lo = byte & 0xF
    hi = (byte >> 4) & 0xF
    idx = torch.stack([lo, hi], dim=1).reshape(K, N).to(torch.int64)
    lvl = inputs["c1"][idx]

    w = lvl * am.repeat_interleave(G, dim=0)
    return {"am": am.contiguous(), "w": w.contiguous()}


SPEC = KernelSpec(
    name="nf4_codebook_dequant_kernel", kernel=kernel,
    signature={"wq_ptr": "*i32", "aq_ptr": "*i32", "c1_ptr": "*fp32",
               "c2_ptr": "*fp32", "s2_ptr": "*fp32", "off_ptr": "*fp32",
               "am_ptr": "*fp32", "w_ptr": "*fp32", "K": "constexpr",
               "N": "constexpr", "G": "constexpr", "NB": "constexpr"},
    constexprs={"K": K, "N": N, "G": G, "NB": NB},
    args=[Arg("wq", IN, "int32", (K // 2, N)),
          Arg("aq", IN, "int32", (NB, N)),
          Arg("c1", IN, "float32", (16,)),
          Arg("c2", IN, "float32", (16,)),
          Arg("s2", IN, "float32", (N,)),
          Arg("off", IN, "float32", (N,)),
          Arg("am", OUT, "float32", (NB, N)),
          Arg("w", OUT, "float32", (K, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-6,
)
