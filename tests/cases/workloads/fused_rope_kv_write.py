"""RoPE applied to q and k, then k and v written into the paged cache -- one pass.

torch equivalent, for one decoded token:
    q, k = rope(q, pos), rope(k, pos)
    kcache[slot] = k
    vcache[slot] = v

WHY IT IS FUSED IN EVERY SERVING STACK
  These three touch the same tiny tensors -- one head's q, k, v for ONE token --
  and separately each is a launch that reads and writes a few hundred bytes. At
  one decoded token per step the launch overhead is the cost, not the
  arithmetic, so vLLM and FlashInfer emit rotary-and-cache as a single kernel.
  It also removes the temporary: the rotated k is never materialised anywhere
  except on its way into the cache.

WHAT THE BACKEND HAS TO DO AT ONCE
  A PERMUTED READ. RoPE pairs dimension j with j + D/2 and mixes them by a
  rotation, so the tile is read twice at an offset of half its width and
  recombined. Nothing about that is elementwise in the index.

  A DATA-DEPENDENT WRITE. The destination row is `slot`, loaded from memory, so
  the store address is not affine -- the same indirection dl/kv_cache_write.py
  isolates, here downstream of the rotation rather than on a plain value.

  BOTH HALVES OF ONE ROTATION LIVE ACROSS THE STORE. q is returned, k is stored,
  and they share the cos/sin tile, so the sin/cos load has two consumers going
  to different places.

HALF-ROTATION LAYOUT (j paired with j+D/2), which is what LLaMA, Qwen and
Mistral use. GPT-NeoX interleaves instead; dl/rope.py holds that distinction.

The caches are IN-OUT: every slot but `slot` must come back untouched, so a store that lands on the wrong row fails on the rows it should not have written as well as the one it should have.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

SLOTS, D = 16, 32                       # cache slots, head dim (even)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fused_rope_kv_write_kernel(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr,
                                   slot_ptr, qo_ptr, kc_ptr, vc_ptr,
                                   D: tl.constexpr):
        half: tl.constexpr = D // 2
        rh = tl.arange(0, half)

        # The rotation reads each half of the tile and mixes them.
        c = tl.load(cos_ptr + rh)
        s = tl.load(sin_ptr + rh)
        q1, q2 = tl.load(q_ptr + rh), tl.load(q_ptr + half + rh)
        k1, k2 = tl.load(k_ptr + rh), tl.load(k_ptr + half + rh)

        qr1, qr2 = q1 * c - q2 * s, q2 * c + q1 * s
        kr1, kr2 = k1 * c - k2 * s, k2 * c + k1 * s

        tl.store(qo_ptr + rh, qr1)
        tl.store(qo_ptr + half + rh, qr2)

        # The destination row is DATA. The rotated k never lands anywhere else.
        slot = tl.load(slot_ptr)
        base = slot * D
        tl.store(kc_ptr + base + rh, kr1)
        tl.store(kc_ptr + base + half + rh, kr2)
        rd = tl.arange(0, D)
        tl.store(vc_ptr + base + rd, tl.load(v_ptr + rd))

    return fused_rope_kv_write_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    half = D // 2
    return {"q": torch.randn(D, generator=g, dtype=torch.float32),
            "k": torch.randn(D, generator=g, dtype=torch.float32),
            "v": torch.randn(D, generator=g, dtype=torch.float32),
            "cos": torch.randn(half, generator=g, dtype=torch.float32),
            "sin": torch.randn(half, generator=g, dtype=torch.float32),
            "slot": torch.tensor([5], dtype=torch.int32),
            "kc": torch.zeros(SLOTS, D, dtype=torch.float32),
            "vc": torch.zeros(SLOTS, D, dtype=torch.float32)}


def reference(inputs):
    import torch
    half = D // 2
    c, s = inputs["cos"], inputs["sin"]

    def rope(x):
        a, b = x[:half], x[half:]
        return torch.cat([a * c - b * s, b * c + a * s])

    slot = int(inputs["slot"][0])
    kc, vc = inputs["kc"].clone(), inputs["vc"].clone()
    kc[slot] = rope(inputs["k"])
    vc[slot] = inputs["v"]
    return {"qo": rope(inputs["q"]), "kc": kc, "vc": vc}


SPEC = KernelSpec(
    name="fused_rope_kv_write_kernel", kernel=kernel,
    signature={"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32",
               "cos_ptr": "*fp32", "sin_ptr": "*fp32", "slot_ptr": "*i32",
               "qo_ptr": "*fp32", "kc_ptr": "*fp32", "vc_ptr": "*fp32",
               "D": "constexpr"},
    constexprs={"D": D},
    args=[Arg("q", IN, "float32", (D,)), Arg("k", IN, "float32", (D,)),
          Arg("v", IN, "float32", (D,)),
          Arg("cos", IN, "float32", (D // 2,)),
          Arg("sin", IN, "float32", (D // 2,)),
          Arg("slot", IN, "int32", (1,)),
          Arg("qo", OUT, "float32", (D,)),
          Arg("kc", OUT, "float32", (SLOTS, D)),
          Arg("vc", OUT, "float32", (SLOTS, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
