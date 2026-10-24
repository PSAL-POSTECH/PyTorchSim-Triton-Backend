"""A reduction result read back over a tile -- the shape softmax is made of.

    s = tl.sum(x)          one value
    y = x - s              every lane needs it

WHY IT IS NOT OBVIOUS THAT THIS WORKS

  Two things stand between the fold and the use, and both are unavailable.
  `tensor.extract` on the rank-0 result puts it in a SCALAR register, and there
  is one scalar register file for all the lanes -- pipeline.py refuses the
  `vfmv.f.s` by name. And the splat that follows wants that one value in every
  lane, which is the move between banks the hardware does not have.

  Neither is needed, because the value can be WRITTEN once and READ many times.
  One element out to DRAM, then a transfer back whose dram stride is ZERO, so
  every position reads the same address and each lane fills its own bank from
  it (torchsim_mvin_common.h:133). That is pytorchsim.relayout with `pytorchsim.replicate`.

  fuse gets to the `linalg.fill` before fit_layout_to_lanes does, so
  what the rewrite actually finds is the scalar CAPTURED in the consumer's
  region, not a fill -- and the capture becomes an ordinary tile operand.

WHAT IT ALSO BUYS

  The relayout CUTS the component. A rank-1 fold answers ONE_LANE, and with
  nothing in between that answer spreads to every buffer it touches: the whole
  kernel would run in lane 0 to serve one reduction. pytorchsim.relayout is in
  lib_axis_graph.NO_EDGE, so the tile side is decided on its own and goes on using
  every lane.

  Both arms are here for that reason. The first is the plain shape; the second
  reduces along the INNER axis of a 2-D tile, where each lane already owns a
  whole row and the result is one value per lane -- an in-lane splat, no round
  trip, and the arrangement to reach for when a kernel can be written that way.

torch equivalent: x - x.sum(), and x2 - x2.sum(1, keepdim=True)

reduce -> broadcast: the rank-1 arm goes out to DRAM and back with a zero stride, the rank-2 one splats inside a lane and does not.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
M = 8
K = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_reduce_broadcast_kernel(x_ptr, o_ptr, p_ptr,
                                    N: tl.constexpr, M: tl.constexpr,
                                    K: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.load(x_ptr + r)
        tl.store(o_ptr + r, x - tl.sum(x, axis=0))      # rank 1: one lane folds

        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        x2 = tl.load(x_ptr + rm[:, None] * K + rk[None, :])
        s2 = tl.sum(x2, axis=1)                          # one value per lane
        tl.store(p_ptr + rm[:, None] * K + rk[None, :], x2 - s2[:, None])

    return ops_reduce_broadcast_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    x2 = x[:M * K].reshape(M, K)
    return {"o": x - x.sum(),
            "p": (x2 - x2.sum(1, keepdim=True)).reshape(-1)}


SPEC = KernelSpec(
    name="ops_reduce_broadcast_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "p_ptr": "*fp32",
               "N": "constexpr", "M": "constexpr", "K": "constexpr"},
    constexprs={"N": N, "M": M, "K": K},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,)),
          Arg("p", OUT, "float32", (M * K,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
