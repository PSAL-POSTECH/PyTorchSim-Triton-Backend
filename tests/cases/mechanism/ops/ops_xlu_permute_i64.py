"""A reverse scan at i64 -- the rotation's wide half.

WHAT IT PINS. `torchsim.crosslane = "permute"` is a lane-to-lane mapping that leaves
the DEPTH alone, and a reverse ladder is what mints one: reading the element
`by` lanes over is a shifted READ, which a rotation answers (ops_scan_reverse
carries that argument). This file changes one thing, the element width, and the
32-bit queues are what then has to be answered.

    before   torchsim-lower-to-vcix: "its element is 64 bits and only 32 crosses
             whole" -- (memref<96xi64>, memref<96xi64>)
    after    ALL STAGES PASSED, max_abs_err 0

HOW A WIDE ROTATION IS TWO ROUNDS, AND WHY EACH REPEATS THE PATTERN. The unit
reads row 0 of what it is handed as `where each lane reads from` and the rows
after it as the tile, and it holds ONE tile -- so the second half is a second
round and needs the pattern pushed again. Both rounds are the same rotation, so
half `h` of lane `k` lands where half `h` of lane `k` should.

WHY THE FIRST ROUND'S ANSWERS SIT IN THE DESTINATION. Keeping them in registers
costs one vector register per pop, and a round is as long as the tile: llc
spills the vector file to the stack and spike answers `VI_ST_WHOLE ERROR:
Accessing invalid spad address`. The destination is somewhere to park, which is
the same reason the all-gather runs its two phases apart.

torch equivalent: x.flip(0).cumsum(0).flip(0), in int64.

A reverse ladder at i64: pins the cross-lane rotation carrying a wide element as its two 32-bit halves, one round each.
"""

#: RE-PINNED TWICE ON 2026-09-23. The old pin held 18 cross-lane passes, and it
#: was ALREADY UNREACHABLE before either re-pin: the kernel stopped at "a subview
#: starts at 1 along the lane axis 0", so 18 is a number from a compiler that no
#: longer exists. One bank was the answer for an hour; now each use of the
#: crossing slice gets its own buffer and the 64 elements stay across 64 lanes,
#: paid for in 12 trip legs. Numbers exact (0 error, TORCHSIM_COMPILE_IGNORE_EXPECT_IR=1).
#:
#: A crossing XLU op carries this permute in one pass, which is what the 12 legs
#: go back to -- the XLU round.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_permute_i64_kernel(x_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        a = tl.load(x_ptr + r)
        tl.store(o_ptr + r, tl.cumsum(a, axis=0, reverse=True))

    return ops_xlu_permute_i64_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-50, 50, (N,), generator=g, dtype=torch.int64)}


def reference(inputs):
    return {"o": inputs["x"].flip(0).cumsum(0).flip(0)}


SPEC = KernelSpec(
    name="ops_xlu_permute_i64_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "o_ptr": "*i64", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "int64", (N,)), Arg("o", OUT, "int64", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # ONE ROTATION IN FRONT OF EACH READER, the way ops_scan_reverse records --
    # six here because this is the sum arm alone.
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane = 16 : i64": 12,
                                   "dram_arg = -1": 0}},
)
