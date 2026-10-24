"""A row intermediate banked on its LAST axis, read by a full-width vector loop.

    pos = t - 3 + k          a [1, L] row, banked with its L across the lanes
    v   = x[c*L + pos]       gathered into a [C, L] tile banked on dim 0

WHAT THIS IS PINNING, AND IT IS NOT THE GATHER. `pos` is a function of `t`
only, so it lives as its own scratchpad buffer of shape [1, L]. fit_layout_to_lanes
banks it on axis 1 -- the L goes ACROSS the lanes, one element in each -- and
place_in_spad tags it `_64lane`, so one lane needs `64 / 64 = 1` element (the [8]
below is from when the reservation rounded that up to MIN_VEC = 8):

    memref.global @bufN_spad_64lane : memref<1x64xi32, 1> {pytorchsim.lane_axis = 1}
    uint32_t bufN_spad_64lane[8] __attribute__ ((section(".spad")));

vectorize then emits the loop that fills it as

    affine.for %i = 0 to 1 {
      affine.for %j = 0 to 64 step 64 {
        affine.vector_load  ...[%i, %j] : memref<1x64xi32, 1>, vector<64xi32>
        affine.vector_store ...[%i, %j] : memref<1x64xi32, 1>, vector<64xi32>

-- 64 elements per lane out of a bank that owns 8. The nest divided dim 0 (the
extent-1 axis) by the lane count and vectorised dim 1, which is the axis the
buffer is actually banked on. The load reads 224 bytes past the buffer and the
store WRITES them, into whichever global the linker put next.

MEASURED, on dl/causal_conv1d, which is this kernel with a filter and a SiLU on
top -- same four taps, same shift. Its `c * L` tile sits directly above the row buffers in `.spad`, so the
third tap's 64-wide store of `t - 3 + 2` landed on it:

    indirect_index2 = c*64 + t + 1        instead of   c*64 + t - 1
    diff exactly +2 on 48 of the 64 columns of every row, 0 on the rest
    (48 = the part of the c*L tile the overrun reached; 3072 of 4096 elements)

read out of the simulator's own `runtime/indirect_access/indirect_index2.raw`,
which is the index buffer as the DMA saw it. Taps 0 and 1 were exact in the same
dump -- the overrun only corrupts when a live buffer happens to be downstream of
it, which is why this is silent rather than reliably fatal.

TWO BUGS MEET HERE AND BOTH ARE FIXED NOW. `pos` is -1 at t = 0, so this also
needs the indirect index tile to be i64 (convert_accesses_to_transfer's
_Ctx.index_buffer) or spike traps before any of the above is reachable -- done in
30b8305. The other half was that vectorize picked its loop nest by asking
whether one vector HAPPENED to cover the extent (`max(MIN_VEC, shape[-1]) >=
shape[0]`) instead of asking which axis the buffer is banked on. It now divides
the LANE axis by the lane count and vectorises one of the others, so for a
[1, L] buffer banked on axis 1 there is nothing left to vectorise but the
extent-1 axis: a one-element loop, which legalize issues at its own width of
one -- exactly the one element a lane needs. "How much of this buffer does one
lane hold" is decided in one place now, off the buffer's `pytorchsim.lane_axis`, and
read in the other.

WHAT MAKES IT FAIL AGAIN: put the width back into the nest choice. Widening the
loop past what the lane axis says the lane holds is the whole bug, and it is a
store, so nothing between here and the memory dump reports it.

torch equivalent: o[c, t] = sum over the last W taps of x[c, t - W + 1 + k],
zero where the tap runs off the front. Reproduced here at W = 4: with W = 2 the
same overrun happens and lands on a buffer that is rewritten afterwards, so the
kernel passes -- it takes a third row for a live neighbour to be downstream of
it, which is exactly why this is silent in the wild.

The smallest form of the overrun measured on dl/causal_conv1d: vectorize vectorised the axis the buffer is BANKED on when that axis is not dim 0, running a vector<L> store 224 bytes past a bank that owns 8. Fixed by choosing the nest off the lane axis instead of off whether one vector happened to cover the extent.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C, L = 64, 64
W = 4                   # four taps -- the third is where the overrun lands


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def lane_axis_row_overrun_kernel(x_ptr, o_ptr, C: tl.constexpr,
                                     L: tl.constexpr, W: tl.constexpr):
        c = tl.arange(0, C)[:, None]
        t = tl.arange(0, L)[None, :]
        acc = tl.zeros((C, L), dtype=tl.float32)
        for k in tl.static_range(W):
            # A row intermediate: a function of t alone, so it becomes its own
            # [1, L] scratchpad buffer banked across the lanes.
            pos = t - (W - 1) + k
            acc += tl.load(x_ptr + c * L + pos,
                           mask=tl.broadcast_to(pos >= 0, (C, L)), other=0.0)
        tl.store(o_ptr + c * L + t, acc)

    return lane_axis_row_overrun_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(C * L, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(C, L)
    o = torch.zeros_like(x)
    for k in range(W):
        sh = W - 1 - k
        o[:, sh:] += x[:, :L - sh] if sh else x
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="lane_axis_row_overrun_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "C": "constexpr",
               "L": "constexpr", "W": "constexpr"},
    constexprs={"C": C, "L": L, "W": W},
    args=[Arg("x", IN, "float32", (C * L,)),
          Arg("o", OUT, "float32", (C * L,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
)
