"""A REBANKING transpose whose tile is LONGER than the machine.

    acc  memref<512x512xf32>   carried, read on axis 0 and written on axis 1
    acc  tl.dot(tl.trans(acc), w)   the transpose is absorbed into the dot's maps

WHAT THIS PINS. `loop_10_carried_axis_swap` is this kernel at 16x16 and is the
suite's ONLY rebanking transpose (`_xlu_rebank`, rung 4): the loop carries a
tile that leaves on one axis and comes back on the other, and nothing free
settles it. At 16 the source rides 16 of the machine's 256 lanes and one
crossbar round covers it. At 512 it rides more positions than there are lanes,
and `_xlu_passes` refused that outright:

    pytorchsim.crosslane = 'transpose' cannot be lowered here (the shapes are not the
    unit's)

S == T's shape ON PURPOSE, for the reason the 16x16 file gives: the accumulator
has to come back the shape it left or the loop does not typecheck.

OPEN, AND NOT FOR THE TRANSPOSE'S SAKE. The block transpose this file was first
written for is built and PROVED elsewhere --
`tile_rebank_longer_than_the_machine` reaches the same rung with no loop at all
and comes back 1.79e-07. What keeps this one open is the back edge, exactly as
in the 16x16 file: fit_layout_to_lanes stamps the copy that closes it with TWO
lane axes and buys nothing to make them agree. Measured at 16, 64, 256 and 512
-- wrong at every size, including the ones with no tall axis.

WHAT IT ADDS, kept for when that closes: the back edge and an axis longer than
the machine in one kernel, which the loop-free file cannot say anything about.

The back edge AND an axis longer than the machine, together. The tall transpose itself is proved by tile_rebank_longer_than_the_machine.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, T = 512, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def transpose_longer_than_the_machine(a_ptr, w_ptr, o_ptr,
                                          S: tl.constexpr, T: tl.constexpr):
        r = tl.arange(0, S)
        acc = tl.load(a_ptr + r[:, None] * S + r[None, :])
        w = tl.load(w_ptr + r[:, None] * S + r[None, :])
        for _ in range(T):
            acc = tl.dot(tl.trans(acc, 1, 0), w)
        tl.store(o_ptr + r[:, None] * S + r[None, :], acc)

    return transpose_longer_than_the_machine


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # 1/sqrt(S), for the reason the 16x16 file gives: each dot sums S products,
    # so anything else grows by sqrt(S) a trip and atol stops meaning anything.
    s = 1.0 / (S ** 0.5)
    return {"a": torch.randn(S * S, generator=g, dtype=torch.float32) * s,
            "w": torch.randn(S * S, generator=g, dtype=torch.float32) * s}


def reference(inputs):
    acc = inputs["a"].reshape(S, S)
    w = inputs["w"].reshape(S, S)
    for _ in range(T):
        acc = acc.transpose(0, 1) @ w
    return {"o": acc.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="transpose_longer_than_the_machine", kernel=kernel,
    signature={"a_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "S": "constexpr", "T": "constexpr"},
    constexprs={"S": S, "T": T},
    args=[Arg("a", IN, "float32", (S * S,)),
          Arg("w", IN, "float32", (S * S,)),
          Arg("o", OUT, "float32", (S * S,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
)
