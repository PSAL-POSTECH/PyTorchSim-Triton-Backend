"""DeepSeek-V2's MLA -- cache a small latent, expand it back at attention time.

    c       = h @ W_dkv                 [T, R]     R << D: this is what is CACHED
    k_nope  = c @ W_uk                  [T, Dn]    expanded per layer, per use
    k_rope  = rotate(h @ W_kr)          [T, Dr]    a SEPARATE, uncompressed path
    k       = cat(k_nope, k_rope)       [T, Dn+Dr]

THE KV CACHE IS THE BOTTLENECK, NOT THE WEIGHTS. Generation is memory bound: at
each step the whole cache is read to produce one token. MHA caches
`2 * n_heads * d_head` numbers per token; GQA cuts that by sharing KV heads
(`gqa_repeat_kv`). MLA instead caches a LOW-RANK PROJECTION and rebuilds k and v
from it -- DeepSeek-V2 reports 93% less cache than its MHA equivalent, and
BETTER quality than MHA, which is the surprising half.

  Why it can be better: the down-then-up projection is a bottleneck the model
  trains through, so it learns what to keep. GQA's sharing is imposed.

WHY RoPE NEEDS ITS OWN PATH -- the detail that makes this an architecture rather
than a compression trick. RoPE rotates by absolute position, so `k_m` and `q_n`
interact through `R(n-m)`. If k came out of an up-projection you would have to
rotate AFTER expanding, which means the expansion cannot be folded into the
query projection ahead of time and the cache saving is given straight back. So a
few dimensions skip the compression entirely and carry the position; the rest
are compressed and carry content. The head dimension is a CONCATENATION of two
differently-derived halves.

WHAT IT ASKS OF THIS MACHINE. Two CHAINED matmuls -- the second reads the
first's result, so the systolic array's MUST lands on both and the latent is an
intermediate that never reaches DRAM. Then a concat along the head dim of two
tiles built by different loop nests, which is `fire_module`'s shape in an LLM.

The rotation is written the way `rope.py` writes it: the two halves are SEPARATE
TILES from the start, produced by two projections against different weight
columns. rotate_half as a shuffle inside one tile would be a lane-crossing move
this machine does not have, and splitting the projection costs nothing because
the weight was going to be read either way.

torch equivalent:
    c = h @ w_dkv
    k = torch.cat([c @ w_uk, rotate_half_apply(h @ w_kr, pos)], dim=-1)

DeepSeek-V2 MLA: a cached low-rank latent, a chained up-projection that reads it, and a head dim CONCATENATED from a compressed half and an uncompressed RoPE half.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, D, R, DN, DR = 64, 64, 16, 32, 32
HALF = DR // 2
BK = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mla_latent_kernel(h_ptr, wd_ptr, wu_ptr, wr_ptr, cos_ptr, sin_ptr,
                          c_ptr, k_ptr,
                          T: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                          DN: tl.constexpr, DR: tl.constexpr,
                          HALF: tl.constexpr, BK: tl.constexpr):
        t = tl.arange(0, T)
        r = tl.arange(0, R)
        kk = tl.arange(0, BK)

        # DOWN-PROJECTION to the latent. This is the tensor a real MLA caches;
        # everything below rebuilds from it.
        c = tl.zeros((T, R), dtype=tl.float32)
        for k0 in range(0, D, BK):
            hk = tl.load(h_ptr + t[:, None] * D + (k0 + kk)[None, :])
            wk = tl.load(wd_ptr + (k0 + kk)[:, None] * R + r[None, :])
            c += tl.dot(hk, wk)
        tl.store(c_ptr + t[:, None] * R + r[None, :], c)

        # UP-PROJECTION: reads the latent the line above just produced.
        n = tl.arange(0, DN)
        kn = tl.zeros((T, DN), dtype=tl.float32)
        for k0 in range(0, R, BK):
            ck = tl.load(c_ptr + t[:, None] * R + (k0 + kk)[None, :])
            wk = tl.load(wu_ptr + (k0 + kk)[:, None] * DN + n[None, :])
            kn += tl.dot(ck, wk)
        tl.store(k_ptr + t[:, None] * (DN + DR) + n[None, :], kn)

        # THE DECOUPLED RoPE HALF, projected from h and not from the latent.
        # Two projections rather than one and a shuffle: the halves are what
        # the rotation mixes, so they are separate tiles from the start.
        v = tl.arange(0, HALF)
        a1 = tl.zeros((T, HALF), dtype=tl.float32)
        a2 = tl.zeros((T, HALF), dtype=tl.float32)
        for k0 in range(0, D, BK):
            hk = tl.load(h_ptr + t[:, None] * D + (k0 + kk)[None, :])
            a1 += tl.dot(hk, tl.load(wr_ptr + (k0 + kk)[:, None] * DR
                                     + v[None, :]))
            a2 += tl.dot(hk, tl.load(wr_ptr + (k0 + kk)[:, None] * DR
                                     + (HALF + v)[None, :]))
        cs = tl.load(cos_ptr + t[:, None] * HALF + v[None, :])
        sn = tl.load(sin_ptr + t[:, None] * HALF + v[None, :])

        # CONCAT along the head dim: the compressed half, then the rotated one.
        base = t[:, None] * (DN + DR) + DN
        tl.store(k_ptr + base + v[None, :], a1 * cs - a2 * sn)
        tl.store(k_ptr + base + (HALF + v)[None, :], a2 * cs + a1 * sn)

    return mla_latent_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(T).float()[:, None]
    inv = torch.exp(-torch.arange(HALF).float() * (2.0 / HALF)
                    * 2.302585092994046)[None, :]
    ang = pos * inv
    return {"h": torch.randn(T * D, generator=g, dtype=torch.float32) * 0.2,
            "wd": torch.randn(D * R, generator=g, dtype=torch.float32) * 0.2,
            "wu": torch.randn(R * DN, generator=g, dtype=torch.float32) * 0.2,
            "wr": torch.randn(D * DR, generator=g, dtype=torch.float32) * 0.2,
            "cos": ang.cos().reshape(-1).contiguous(),
            "sin": ang.sin().reshape(-1).contiguous()}


def reference(inputs):
    import torch
    h = inputs["h"].reshape(T, D)
    c = h @ inputs["wd"].reshape(D, R)
    kn = c @ inputs["wu"].reshape(R, DN)
    a = h @ inputs["wr"].reshape(D, DR)
    a1, a2 = a[:, :HALF], a[:, HALF:]
    cs = inputs["cos"].reshape(T, HALF)
    sn = inputs["sin"].reshape(T, HALF)
    kr = torch.cat([a1 * cs - a2 * sn, a2 * cs + a1 * sn], dim=1)
    return {"c": c.reshape(-1).contiguous(),
            "k": torch.cat([kn, kr], dim=1).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="mla_latent_kernel",
    kernel=kernel,
    signature={"h_ptr": "*fp32", "wd_ptr": "*fp32", "wu_ptr": "*fp32",
               "wr_ptr": "*fp32", "cos_ptr": "*fp32", "sin_ptr": "*fp32",
               "c_ptr": "*fp32", "k_ptr": "*fp32",
               "T": "constexpr", "D": "constexpr", "R": "constexpr",
               "DN": "constexpr", "DR": "constexpr", "HALF": "constexpr",
               "BK": "constexpr"},
    constexprs={"T": T, "D": D, "R": R, "DN": DN, "DR": DR, "HALF": HALF,
                "BK": BK},
    args=[Arg("h", IN, "float32", (T * D,)),
          Arg("wd", IN, "float32", (D * R,)),
          Arg("wu", IN, "float32", (R * DN,)),
          Arg("wr", IN, "float32", (D * DR,)),
          Arg("cos", IN, "float32", (T * HALF,)),
          Arg("sin", IN, "float32", (T * HALF,)),
          Arg("c", OUT, "float32", (T * R,)),
          Arg("k", OUT, "float32", (T * (DN + DR),))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
