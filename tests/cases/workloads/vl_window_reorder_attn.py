"""Qwen2.5-VL / Qwen3-VL vision window attention -- gather, attend, SCATTER BACK.

    x_perm = x[window_index]          patches reordered into contiguous windows
    attend inside each window only
    out[window_index] = attended      the inverse permutation, as a scatter

WHY THE PATCHES HAVE TO MOVE AT ALL. The ViT sees a variable-resolution image as
a flat run of patches in raster order, so a spatial window -- 8x8 patches in
Qwen2.5-VL -- is a set of rows that are STRIDED, not contiguous: it takes one
short segment per patch-row, and where those segments are depends on the image's
width, which changes per image. Attention wants each window to be one contiguous
run so the whole batch of windows can go through a single varlen flash kernel
with one `cu_seqlens`. So `get_window_index` builds the permutation ONCE on the
host, the patches are gathered into window order, attention runs, and the
result is put back where it came from.

  Only the windowed layers do this. Qwen2.5-VL runs full attention on 4 of its
  32 vision blocks and windowed attention on the other 28, and the permute /
  unpermute wraps the whole windowed stack rather than each block.

THE UNPERMUTE IS THE SAME INDEX TENSOR, USED AS A DESTINATION. The reference
implementation builds `reverse_indices = window_index.argsort()` and indexes
with it; `out[window_index[i]] = attended[i]` is the identical map with no
second tensor and no sort, which is what this kernel does. A gather and a
scatter through ONE index array, with a reduction in between -- the address
family is read twice, once on each side of a softmax.

  `swin_window.py` is the other windowed attention here and it is structurally
  the opposite: its window map is `(y + S) mod H`, pure arithmetic each lane
  does for itself, with NO INDEX TENSOR ANYWHERE. This one's map is DRAM. So
  Swin exercises non-affine address arithmetic and this one exercises the
  row-indirect DMA -- `embedding.py`'s `indirect_mode`, one index per lane --
  and then asks for it in the STORE direction too, which nothing else here
  does.

WHAT IS PINNED AND WHY. The real thing has a data-dependent number of windows
and a ragged last window per image (Qwen2.5-VL pads the grid up to the window
edge and masks the pad away). Shapes here are static, so the window count and
the window size are constants and the ONLY data is the permutation itself --
which is the part being tested. A ragged tail would be `softmax_causal.py`'s
`other=` masking bolted on and would test that instead.

torch equivalent:
    xp = x[widx].view(NW, WS, D)
    o  = torch.empty_like(x)
    o[widx] = softmax(qp @ kp.mT * scale) @ vp

Qwen2.5-VL: a row-indirect gather into window order, a softmax, and the SAME index tensor used as a scatter destination to undo it -- the indirect DMA in both directions around a reduction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NW = 4                     # windows
WS = 16                    # patches per window == the systolic edge
T = NW * WS                # patches in the image
D = 16                     # head dim
SCALE = 1.0 / (D ** 0.5)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def vl_window_reorder_attn_kernel(widx_ptr, q_ptr, k_ptr, v_ptr, o_ptr,
                                      NW: tl.constexpr, WS: tl.constexpr,
                                      D: tl.constexpr, SCALE: tl.constexpr):
        w = tl.arange(0, WS)
        d = tl.arange(0, D)

        # The window axis is a static loop: putting it in the tile would make
        # every tile rank 3, and NW is a constant here for the reason the
        # docstring gives.
        for g in tl.static_range(NW):
            # THE PERMUTATION, from DRAM. One index per lane, and the same
            # index family serves all three of q, k, v.
            idx = tl.load(widx_ptr + g * WS + w)
            base = idx[:, None] * D + d[None, :]

            q = tl.load(q_ptr + base)
            k = tl.load(k_ptr + base)
            v = tl.load(v_ptr + base)

            s = tl.dot(q, tl.trans(k)) * SCALE
            m = tl.max(s, axis=1)
            e = tl.exp(s - m[:, None])
            p = e / tl.sum(e, axis=1)[:, None]

            # THE UNPERMUTE: the gather's own addresses, used as a destination.
            tl.store(o_ptr + base, tl.dot(p, v))

    return vl_window_reorder_attn_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # A real permutation, unsorted on purpose: a window whose indices happened
    # to be contiguous would be indistinguishable from a plain slice.
    widx = torch.randperm(T, generator=g).to(torch.int32)
    return {"widx": widx,
            "q": torch.randn(T * D, generator=g, dtype=torch.float32) * 0.5,
            "k": torch.randn(T * D, generator=g, dtype=torch.float32) * 0.5,
            "v": torch.randn(T * D, generator=g, dtype=torch.float32) * 0.5}


def reference(inputs):
    import torch
    widx = inputs["widx"].long()
    q = inputs["q"].reshape(T, D)
    k = inputs["k"].reshape(T, D)
    v = inputs["v"].reshape(T, D)
    o = torch.zeros(T, D)
    for g in range(NW):
        idx = widx[g * WS:(g + 1) * WS]
        p = torch.softmax(q[idx] @ k[idx].T * SCALE, dim=-1)
        o[idx] = p @ v[idx]
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="vl_window_reorder_attn_kernel",
    kernel=kernel,
    signature={"widx_ptr": "*i32", "q_ptr": "*fp32", "k_ptr": "*fp32",
               "v_ptr": "*fp32", "o_ptr": "*fp32", "NW": "constexpr",
               "WS": "constexpr", "D": "constexpr", "SCALE": "constexpr"},
    constexprs={"NW": NW, "WS": WS, "D": D, "SCALE": SCALE},
    args=[Arg("widx", IN, "int32", (T,)),
          Arg("q", IN, "float32", (T * D,)),
          Arg("k", IN, "float32", (T * D,)),
          Arg("v", IN, "float32", (T * D,)),
          Arg("o", OUT, "float32", (T * D,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
