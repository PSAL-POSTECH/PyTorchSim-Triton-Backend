"""A pointer bitcast that NARROWS the element, crossed by rescaling the offsets.

    pb = tl.cast(x_ptr + r, tl.pointer_type(tl.int8))   # f32* -> i8*, 4 -> 1
    tl.load(pb + i)                                     # i is a runtime index

THE OFFSET IS ALREADY ON THE POINTER when the cast happens, and that is the
whole point of the file. Casting the bare argument is FREE: nothing has
accumulated, so the new element size is simply the one every later offset is
counted in. It is `x_ptr + r` -- `r` counted in f32 elements, then reinterpreted
as bytes -- that has to be RESCALED, because `tts.make_tptr`'s offsets and
strides are ELEMENT counts and the element just changed.

    02-ttclean    tt.bitcast tensor<32x!tt.ptr<f32>> -> tensor<32x!tt.ptr<i8>>
    03-linalg   tts.make_gather_scatter_tptr, gather offset 4*r + i, on <i8>

WHY THE DESCRIPTOR CAN HOLD IT AFTER ALL, which this file used to deny. The
descriptor has ONE element_size and it comes from the TILE, never from the DRAM
argument (lib_descriptor.TransferFacts reads the SRAM tile's type and hands
that one number to both the DRAM address and the SPAD address). So an i8 tile
makes the DRAM grid a BYTE grid, both sides of the transfer are one byte wide,
and no second width is needed. `memref<*xf32>` on the argument is a name that
never enters an address -- convert_accesses_to_transfer's base_memref has said so all along.

WHAT THE WALK DOES NOW. PtrAnalysis::visitOperandBitcast asks
Utils::ptrCastElementRatio for how many DST elements one SRC element spans and
multiplies the whole address expression by it: offsets, strides, the per-axis
wrap, baseOffset, flatShape. `(x mod W)*a == (x*a) mod (W*a)`, the identity
mulState already uses. Here that turns stride 1 into stride 4, and the gather
offset that finally reaches the descriptor is `4*r + i` -- the bytes this
kernel's reference() reads.

TWO THINGS IT STILL REFUSES, both pinned in
test/Conversion/TritonToStructured/bitcast_keeps_the_chain.mlir:

    widening (i8* -> i32*)          recovering the index is a DIVISION and
        nothing proves the accumulated offset is a multiple. Census below says
        this never happens; refused rather than guessed.
    a GATHERED dim at the cast      that dim's contribution is
        offsets[d] * strides[d], so scaling both scales it twice -- and
        offsets[d] is a per-element tensor the scalar factor cannot multiply at
        all. MEASURED by removing the guard: `arith.muli(tensor<128xi32>,
        index)` fails the verifier.

WHERE THE ARGUMENT'S WIDTH DOES STILL DECIDE SOMETHING, because it is not
nowhere -- it is just not in _tile_elem, which used to compare against it:

    convert_accesses_to_transfer _emit_scalar          a scalar access is a memref.load on the flat view,
        whose element type IS the argument's, and there is no descriptor to
        scale. A width mismatch is refused THERE.
    convert_accesses_to_transfer _park_masked_offsets  how big the buffer is is counted in the ARGUMENT's
        element; where the park lands is counted in the TILE's. Bytes is the one
        unit both agree on, and four of its five callers were already handing it
        the tile's element under a parameter named `arg_elem`.

Census over the suite's 410 kernels: eleven pointer bitcasts, `f32->i32` five
times, `i1->i8` five, and this one narrowing. Widening: none. Real model IR
(PyTorchSim/outputs, nine dumps): no pointer bitcast at all.

WHAT BREAKS IT

    crossing a width-changing cast WITHOUT the rescale    the offset is carried
        across a 4x element-size change and every address moves. This kernel
        answers with x's bytes at 4*k + i; an unscaled crossing answers with the
        bytes at k + i.
    comparing bit width instead of storage bytes           i1 is one BIT and one
        BYTE. MEASURED by making the predicate compare getIntOrFloatBitWidth:
        396/398, and the two that fall are ops_cumsum_bool_promoted and
        ptr_bitcast_bool_scalar_load. I had written "all five bool bitcasts"
        here; five is how many bitcasts there are, two is how many kernels
        notice, and the difference is that the other three reach the same
        answer down the elementwise path anyway.
    dropping the max(8u, ...) in storageBytes              nothing. divideCeil
        already rounds a sub-byte width up to one, so max(8u, n)/8 and
        ceil(n/8) agree at EVERY width -- measured, 398/398 unchanged. The max
        states the intent and buys no behaviour; do not read it as the guard.

torch equivalent: x reinterpreted as int8, read at byte 4*k + i[k].

The one pointer bitcast in the suite that NARROWS the element. No descriptor can hold it -- offsets are element counts and there is one element size -- so both walks refuse to cross it and the elementwise path computes the byte address instead. The right answer here is the evidence that refusing is not losing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def bitcast_narrows_the_element(x_ptr, i_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(i_ptr + r)
        # THE OFFSET IS ALREADY ON THE POINTER when the cast happens: `r` was
        # counted in f32 elements, and after the cast the same number means
        # BYTES. Casting the bare argument would be harmless -- nothing has
        # accumulated yet -- which is why this one adds `r` first.
        pb = tl.cast(x_ptr + r, tl.pointer_type(tl.int8))
        tl.store(o_ptr + r, tl.load(pb + i))

    return bitcast_narrows_the_element


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BLOCK * 4, generator=g, dtype=torch.float32),
            "i": torch.randint(0, BLOCK, (BLOCK,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    """The byte at 4*k + i[k] of x, read as int8.

    `x_ptr + r` steps f32 ELEMENTS, so position k starts at byte 4*k. The cast
    renames the pointee without moving it, and `+ i` then steps BYTES."""
    import torch
    b = inputs["x"].view(torch.int8)
    i = inputs["i"].to(torch.int64)
    k = torch.arange(BLOCK, dtype=torch.int64)
    return {"o": b[4 * k + i].contiguous()}


SPEC = KernelSpec(
    name="bitcast_narrows_the_element", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*i8",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (BLOCK * 4,)),
          Arg("i", IN, "int32", (BLOCK,)),
          Arg("o", OUT, "int8", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
