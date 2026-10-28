"""yolov5's SPPF maxpool, Inductor's kernel as it ran: a store through a computed index, masked, on a cut dim.

WHAT IT IS. The maxpool reduces a 5 x 5 window of a [XBLOCK, R0_BLOCK] tile and writes the [XBLOCK, 1] result
into a concatenated buffer at `x0 + 512 * x6` under the tail mask `x < 51200`, a block of 65536. The address is no
stride and the mask keeps the store indexed along dim 0, which the layout cuts into lane and register digits.

WHAT IT USED TO DO. Two refusals, one behind the other:

    DigitRefused: a DMA masked_axes on a dim cut into digits [128, 512] whose bound is not constant
    DigitRefused: a gathering DMA whose gathered dim is cut into digits

The tail bound min(max(min(pid*65536 + 65536, 51200), pid*65536) - pid*65536, 65536) is a multiple of 512, the
cut dim's inner run, so it is one box on the outer digit. Every other dim of the data tile is 1, so its index has
one entry per element: the transfer reads its index buffer per element (no indirect_dim), laid as the data tile.

WHAT WOULD BREAK THIS
  * keeping indirect_dim on the cut dim -- the refusal above
  * keeping a runtime bound on a cut dim refused -- the first refusal above
  * no mask -- the index is then per element from the start and never had an indirect_dim

torch equivalent: o[x0 + 512 * x6] = max over the 5 x 5 window of a, the window's taps outside 20 x 20 skipped
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, XN, XB = 204416, 51200, 65536


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_scatter_store_across_cut_rows(in_ptr0, out_ptr0, XBLOCK: tl.constexpr):
        R0_BLOCK: tl.constexpr = 32
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < 51200
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_mask = r0_index < 25
        r0_4 = r0_index // 5
        x2 = xindex // 2560
        r0_3 = r0_index % 5
        x1 = (xindex // 128) % 20
        x0 = xindex % 128
        x6 = xindex // 128
        tmp0 = (-2 + r0_4 + x2).to(tl.int32)
        tmp6 = (-2 + r0_3 + x1).to(tl.int32)
        ok = (tmp0 >= 0) & (tmp0 < 20) & (tmp6 >= 0) & (tmp6 < 20)
        v = tl.load(in_ptr0 + (-21504 + x0 + 512 * r0_3 + 512 * x6 + 10240 * r0_4), r0_mask & ok & xmask,
                    other=float("-inf")).to(tl.float32)
        m = tl.max(tl.where(r0_mask, v, float("-inf")), 1)[:, None]
        tl.store(out_ptr0 + (x0 + 512 * x6), m.to(tl.bfloat16), xmask)
    return gather_scatter_store_across_cut_rows


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(N, generator=g).to(torch.bfloat16),
            "o": torch.zeros(N, dtype=torch.bfloat16)}


def reference(inputs):
    import torch
    a, o = inputs["a"].float(), inputs["o"].clone()
    x = torch.arange(XN)
    x0, x1, x2, x6 = x % 128, (x // 128) % 20, x // 2560, x // 128
    best = torch.full((XN,), float("-inf"))
    for r in range(25):
        h, w = x2 - 2 + r // 5, x1 - 2 + r % 5
        ok = (h >= 0) & (h < 20) & (w >= 0) & (w < 20)
        addr = (-21504 + x0 + 512 * (r % 5) + 512 * x6 + 10240 * (r // 5)).clamp(0, N - 1)
        best = torch.maximum(best, torch.where(ok, a[addr], torch.tensor(float("-inf"))))
    o[x0 + 512 * x6] = best.to(torch.bfloat16)
    return {"o": o}


SPEC = KernelSpec(
    name="gather_scatter_store_across_cut_rows", kernel=kernel,
    signature={"in_ptr0": "*bf16", "out_ptr0": "*bf16", "XBLOCK": "constexpr"},
    constexprs={"XBLOCK": XB},
    args=[Arg("a", IN, "bfloat16", (N,)),
          Arg("o", "inout", "bfloat16", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
