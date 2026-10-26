"""A staged copy that a TURN makes into a permutation -- the one way a tile
shuffle reaches fit_layout_to_lanes, and it is not a `tl.trans`.

WHAT IT PINS
`fit_layout_to_lanes._moves_the_tile`, the permutation half of
`_shuffles_the_tile`, and the `pytorchsim.perm` the round trip carries: the MVOUT
SCATTERS the source into the destination's image (input axis a takes the stride
of the output axis the permutation sent it to) while the MVIN stays an ordinary
read. Sizing that MVOUT's strides by the OUTPUT is right only while the two
shapes agree, which is exactly what a permutation breaks.

WHY IT IS NOT A `tl.trans`, MEASURED. No `linalg.transpose` survives
absorb_layout_ops anywhere in the suite -- 0 of 435 kernels -- so a Triton-level
transpose is absorbed into a `dram_stride` and never reaches this pass. The op
that does reach it is one fit_layout_to_lanes MINTS: an identity copy staged for
a row, whose two maps a later turn permutes into

    ins  (d0, d1) -> (d0, d1)      memref<64x4xf32>
    outs (d0, d1) -> (d1, d0)      memref<4x64xf32>

so the arm can only be reached by setting up the conflict that makes the pass
stage AND turn, not by writing a transpose.

INGREDIENTS, BY LEAVE-ONE-OUT. The RING WRITE is load-bearing: with the masked
store deleted the same kernel reaches nothing and carries 0 round trips. Four
reductions were tried before this one and none reached the arm -- a transpose
with two readers, a transpose feeding a reassociation-free reshape, both of
those over a loop-carried tile, and this kernel without its ring.

WHERE IT CAME FROM. Reduced from workloads/causal_conv1d_update.py, which was
the ONLY kernel in the suite reaching this arm; ops_relayout_shapes.py was
written for it and stopped reaching it once absorb_layout_ops took its
transpose. The convolution is gone and the ring is what is left.

Reaches fit_layout_to_lanes._moves_the_tile and the pytorchsim.perm MVOUT, which ops_relayout_shapes.py
stopped reaching. The op is one the pass mints and turns, not a tl.trans.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import (KernelSpec, Arg, IN,
                                                        OUT, INOUT)

D, W, WM1 = 64, 4, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_turned_copy_is_a_shuffle_kernel(x_ptr, w_ptr, s_ptr, o_ptr,
                                            D: tl.constexpr, W: tl.constexpr,
                                            WM1: tl.constexpr):
        d = tl.arange(0, D)[:, None]
        k = tl.arange(0, W)[None, :]
        rd = tl.arange(0, D)
        x = tl.load(x_ptr + rd)
        st = tl.load(s_ptr + d * WM1 + k,
                     mask=tl.broadcast_to(k < WM1, (D, W)), other=0.0)
        xb = tl.broadcast_to(x[:, None], (D, W))
        acc = tl.sum(tl.where(k == WM1, xb, st) * tl.load(w_ptr + d * W + k),
                     axis=1)
        tl.store(o_ptr + rd, acc)
        # THE RING WRITE, and it is what reaches the arm: a masked store that
        # re-reads the buffer it writes, so the row has to be staged and turned.
        nxt = tl.load(s_ptr + d * WM1 + (k + 1),
                      mask=tl.broadcast_to(k + 1 < WM1, (D, W)), other=0.0)
        tl.store(s_ptr + d * WM1 + k, tl.where(k + 1 == WM1, xb, nxt),
                 mask=tl.broadcast_to(k < WM1, (D, W)))

    return ops_turned_copy_is_a_shuffle_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(D, generator=g, dtype=torch.float32),
            "w": torch.randn(D * W, generator=g, dtype=torch.float32),
            "s": torch.randn(D * WM1, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x, w = inputs["x"], inputs["w"].reshape(D, W)
    st = inputs["s"].reshape(D, WM1)
    win = torch.cat([st, x[:, None]], dim=1)
    nxt = torch.cat([st[:, 1:], x[:, None]], dim=1)
    return {"o": (win * w).sum(dim=1), "s": nxt.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="ops_turned_copy_is_a_shuffle_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "s_ptr": "*fp32",
               "o_ptr": "*fp32", "D": "constexpr", "W": "constexpr",
               "WM1": "constexpr"},
    constexprs={"D": D, "W": W, "WM1": WM1},
    args=[Arg("x", IN, "float32", (D,)), Arg("w", IN, "float32", (D * W,)),
          Arg("s", INOUT, "float32", (D * WM1,)),
          Arg("o", OUT, "float32", (D,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
