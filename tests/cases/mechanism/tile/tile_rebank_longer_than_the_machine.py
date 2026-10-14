"""A tile read on BOTH its axes at once, longer than the machine.

    c    tl.dot(a, b)          (S, S), the array pins it on axis 1
    o    c * tl.trans(c)       the same value read (d0,d1) AND (d1,d0)

WHAT THIS PINS. `tl.trans` is absorbed into the generic's `indexing_maps`
(rule 8), so one elementwise op reads ONE value through two maps that name
different iteration dims. No free rung settles that -- the value cannot ride
both -- so fit_layout_to_lanes rebanks one side with the crossbar's TRANSPOSE
(`_xlu_rebank`, rung 4). At S = 512 the tile rides more positions than the
machine has lanes, so that transpose is a BLOCK transpose: `_xlu_groups` counts
the source's lane-fulls, `_xlu_passes` the destination's, and block (a, b) of
one is block (b, a) of the other.

WHY NOT `loop_10_carried_axis_swap`, which is the other rebanking site: it is
`expect="open"` for an unrelated reason -- the copy that closes its back edge is
stamped with two lane axes and nothing pays for the equation -- so it is wrong
at every size and can prove nothing about the transpose. This file has no loop.

WHAT WOULD BREAK THIS. Refusing `s_ext > lanes` in `_xlu_passes` again, or
walking the source's blocks in the wrong order: the DMA stages a lane axis ONE
LANE-FULL AT A TIME (`VLANE_STRIDE = 1`, one lane-axis element per bank), so
block a of the source is positions [a*lanes, (a+1)*lanes) and its transpose is
the slice [a*lanes, (a+1)*lanes) of EVERY destination row -- not a whole half
of them.

512 over 256 lanes on a rank-2 tile, with no loop: the rebanking transpose is the one crossbar operation whose SOURCE can be longer than the machine, and this is where that is measured.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, K = 512, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rebank_longer_than_the_machine(a_ptr, b_ptr, o_ptr,
                                       S: tl.constexpr, K: tl.constexpr):
        r = tl.arange(0, S)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + r[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * S + r[None, :])
        c = tl.dot(a, b)
        # BOTH AXES AT ONCE. The transpose folds into this op's maps, so the one
        # value is read on (d0,d1) and on (d1,d0) in the same nest.
        o = c * tl.trans(c, 1, 0)
        tl.store(o_ptr + r[:, None] * S + r[None, :], o)

    return rebank_longer_than_the_machine


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = 1.0 / (K ** 0.5)
    return {"a": torch.randn(S * K, generator=g, dtype=torch.float32) * s,
            "b": torch.randn(K * S, generator=g, dtype=torch.float32) * s}


def reference(inputs):
    c = inputs["a"].reshape(S, K) @ inputs["b"].reshape(K, S)
    return {"o": (c * c.transpose(0, 1)).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="rebank_longer_than_the_machine", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "K": "constexpr"},
    constexprs={"S": S, "K": K},
    args=[Arg("a", IN, "float32", (S * K,)),
          Arg("b", IN, "float32", (K * S,)),
          Arg("o", OUT, "float32", (S * S,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
