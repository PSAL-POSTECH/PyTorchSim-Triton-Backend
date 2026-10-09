"""A COMPUTED row of i64 that every lane needs whole -- the broadcast's wide half.

WHAT IT PINS. `torchsim.crosslane = "broadcast"` is lane 0's row handed to every
lane, and the queues between them carry 32 bits. A 64-bit element used to be
refused by name here; now it crosses twice, low halves then high, and the far
side puts the two back together. The two halves take the SAME replication, which
is the whole reason they may be split at all.

    before   torchsim-lower-to-vcix: "its element is 64 bits and only 32 crosses
             whole" -- (memref<4xi64>, memref<64x4xi64>)
    after    ALL STAGES PASSED, max_abs_err 0

WHY THE ROW IS A CUMSUM AND NOT A LOAD. A LOADED row costs nothing to replicate:
`_clone_loaded_row_wide` re-reads the same DRAM with a zero stride on the
broadcast axis, so the operand never reaches the unit. A computed row has no
descriptor to widen, which is why `_broadcast_by_a_replicating_trip` exists and
why Pixtral's block-diagonal mask (block_diag_attn_mask.py) is the workload
that first needed it. Replace the `tl.cumsum` with a plain load and this kernel
passes on either tree, pinning nothing.

WHY THE ROW IS SHORT AND THE TILE IS TALL. The lanes ride the 64-axis and the
row is the 4-axis, so every one of the 64 lanes needs all four values. Swap the
extents and the broadcast is an in-lane splat with nothing to move.

torch equivalent: x.reshape(64, 4) + cumsum(c)[None, :]

A computed i64 row replicated to every lane: pins the cross-lane broadcast carrying a wide element as its two 32-bit halves.
"""

#: RE-PINNED TWICE ON 2026-09-23, and the second one is the route that KEEPS
#: THE LANES. The old pin held six cross-lane passes. In between, the slice arm
#: answered a lane-crossing slice with one bank and the count fell to one pass;
#: now `LowerCrossingSlices` gives each such use its own buffer and the tile
#: stays spread, paid for in eight trip legs. Numbers exact (0 error).
#:
#: THE TRIPS ARE WHAT THE CROSS-LANE UNIT WOULD TAKE BACK: a crossing XLU op
#: carries this shift in one pass, and `wanted_pairs` cannot pair a slice with
#: its source because their element counts differ. That is the XLU round.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, NI = 64, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_broadcast_i64_kernel(c_ptr, x_ptr, o_ptr, S: tl.constexpr,
                                     NI: tl.constexpr):
        i = tl.arange(0, S)
        j = tl.arange(0, NI)
        w = tl.cumsum(tl.load(c_ptr + j), axis=0)
        x = tl.load(x_ptr + i[:, None] * NI + j[None, :])
        tl.store(o_ptr + i[:, None] * NI + j[None, :], x + w[None, :])

    return ops_xlu_broadcast_i64_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"c": torch.randint(0, 5, (NI,), generator=g, dtype=torch.int64),
            "x": torch.randint(-50, 50, (S * NI,), generator=g,
                               dtype=torch.int64)}


def reference(inputs):
    w = inputs["c"].cumsum(0)
    return {"o": inputs["x"].reshape(S, NI) + w[None, :]}


SPEC = KernelSpec(
    name="ops_xlu_broadcast_i64_kernel", kernel=kernel,
    signature={"c_ptr": "*i64", "x_ptr": "*i64", "o_ptr": "*i64",
               "S": "constexpr", "NI": "constexpr"},
    constexprs={"S": S, "NI": NI},
    args=[Arg("c", IN, "int64", (NI,)), Arg("x", IN, "int64", (S * NI,)),
          Arg("o", OUT, "int64", (S, NI))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # THE MOVES ARE THE CLAIM -- a loaded row would be widened in its transfer
    # instead and nothing would cross.
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane = 16 : i64": 4,
                                   "dram_arg = -1": 0}},
)
