"""A tile that genuinely does not fit, refused with the marker a CALLER reads.

`tl.sum(x * y * z, axis=1)` over 16 rows of 65536 f32 keeps THREE loaded tiles
of 262144 bytes per lane live, against a double-buffer budget of half the
scratchpad. It is over on the TOTAL, with no single buffer to blame, which is the
other shape of scratchpad failure from `tile_spad_unit_axis.py`, where one buffer
was mis-sized and the total was incidental.

THE SHAPE IS CHOSEN FOR THE RATIO, NOT THE MARGIN. The exact numbers move with
SPAD_SIZE (the budget is half of it, and TORCHSIM_COMPILE_SPAD_SIZE overrides),
so a kernel that is over by a margin stops pinning anything the day the config
changes -- and that happened twice: [128, 8192] cleared a 256-lane budget, and
[32, 32768] with two loads cleared tpuv6e's 1024 KB lanes (the product fused, so
it held two 131072-byte buffers against 524288). Three loads of [16, 65536] are
786432 against 524288, over by 1.5x and under the hard `spad` limit of 1048576,
so it is the SOFT refusal that fires. triton's numel limit is 1048576 and a row
sits on one lane, so fewer rows is the only way to more bytes per lane.

WHAT THIS PINS, AND IT IS NOT THE REFUSAL. That the refusal carries

    torchsim-spad-overflow: usage=<n> budget=<n>

which is `pipeline.SPAD_OVERFLOW_MARKER`, and which exists for exactly one reader:
PyTorchSim's Inductor route, in another process, choosing R0_BLOCK. It has to
choose before it can know how many block-sized tiles the kernel will keep live
-- the count is a property of the lowering and the lowering needs the block size
-- so its only honest move is to compile, be told, and divide. Prose cannot be
divided; `usage / budget` can.

It is the two NUMBERS that make it a contract rather than a flag. "Over budget,
halve it" walks to a fit one step at a time and recompiles at every step; over
by 1.18x is one halving and over by 5x is one eighth, decided at the first
attempt. Measured on ViT's first LayerNorm, which is where this came from:
77504 bytes/lane at R0_BLOCK 512, 38680 at 256, and the retry goes straight
there.

WHAT WOULD BREAK THIS
  * dropping the marker, or renaming either key, or printing them in the other
    order -- the caller stops matching, gives up on a kernel it could have
    compiled, and the failure surfaces as a scratchpad error in a model run
    with no mention of block sizes.
  * routing it through TnpuError's message instead of its raw output: that
    filter keeps only lines that look like a diagnostic, and this line looks
    like none of them ON PURPOSE, because it is addressed to a function.

WHY blocked AND NOT refuse. Nothing here is a decision of this backend's -- the
tile is bigger than the machine, and TORCHSIM_COMPILE_SPAD_SIZE moves the line. It is a fact
about the target, which is what `blocked` means.

Pins the machine-readable half of a scratchpad refusal, which is what PyTorchSim's Inductor route retries R0_BLOCK on.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 16, 65536


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def spad_over_budget(x_ptr, y_ptr, z_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        off = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + off)
        y = tl.load(y_ptr + off)
        z = tl.load(z_ptr + off)
        tl.store(o_ptr + rm, tl.sum(x * y * z, axis=1))

    return spad_over_budget


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32),
            "y": torch.randn(M, N, generator=g, dtype=torch.float32),
            "z": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": (inputs["x"] * inputs["y"] * inputs["z"]).sum(dim=1)}


SPEC = KernelSpec(
    name="spad_over_budget", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "z_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("y", IN, "float32", (M, N)),
          Arg("z", IN, "float32", (M, N)),
          Arg("o", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="blocked",
    expect_why="three 262144-byte/lane tiles against a double-buffer budget of "
               "half the scratchpad: the tile is bigger than the machine, and "
               "the refusal carries pipeline.SPAD_OVERFLOW_MARKER for the caller "
               "that can shrink it",
)
