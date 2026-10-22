"""A gather indexed by an i16, which the linalg lowering's ODS refuses -- and that
refusal is what lets convert_accesses_to_transfer stop widening indices at all.

NEGATIVE KERNEL. It is here to keep failing.

WHAT IT PINS. The DMA descriptor's `indirect_esize` (byte +130) is a BYTE COUNT
and the DMA dispatches on exactly four of them -- `load_int8`, `load_int16`,
`load_int32`, `load_uint64` (torchsim_mvin_common.h) -- with an
`Unsupported index type` assert on anything else. So a gather whose index
element occupies 3 bytes, or 5, could not be described to the hardware at all.

`index_buffer` used to sidestep that by widening EVERY index tile to i64. That
widening is gone (it was covering a spike sign bug and a fit_layout_to_lanes buffer-reuse bug,
both since fixed at their source), and what makes its absence safe is not our
own check but an ODS constraint one level up:

    tts.make_gather_scatter_tptr, operand #1  ->  i32 or i64, nothing else

4 and 8 are both widths `indirect_esize` can name, so nothing needs widening.
This file is the measurement of that constraint rather than a claim about it:

    error: 'tts.make_gather_scatter_tptr' op operand #1 must be tensor of
    32-bit signless integer values or tensor of 64-bit signless integer
    values, but got 'tensor<32xi16>'

WHY A NEGATIVE KERNEL AND NOT A GUARD IN THE PASS. A branch for the widths
upstream already excludes is a branch nothing can reach, nothing can test, and
nobody will notice when it rots -- and this is the function that has already
been burned twice by a safeguard that rode on something else instead of standing
on its own (its docstring names both). The invariant is therefore checked from
OUTSIDE the pass, here, where it fails loudly the day upstream relaxes it. That
day is the day `index_buffer` needs a widening back, and the pass says so in a
`_Fatal` naming this file.

WHAT WOULD BREAK THIS. The linalg lowering widening that ODS to accept i8/i16 indices.
Then this kernel starts PASSING -- which is the signal, not a success -- and
`index_buffer` has to widen the sub-4-byte cases before the descriptor is built,
because `elem_bytes(i8)` is 1 and `elem_bytes(i16)` is 2 and the DMA can name
both, so it would silently read a 2-byte index where the buffer holds one.

SCOPE. Stops at stage 3; the pytorchsim-triton-compiler passes never see it. That is the point --
the refusal is upstream's, and this file is how we notice if it moves.

an i16 gather index, refused upstream. NEGATIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

N = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_index_width_ods_kernel(x_ptr, i_ptr, o_ptr,
                                          N: tl.constexpr):
        r = tl.arange(0, N)
        # THE i16 IS THE WHOLE KERNEL. Loaded rather than computed, because a
        # computed index is promoted to i32 by triton's own type rules before
        # it ever reaches the pointer arithmetic -- reading it from an i16
        # buffer is what carries the narrow type into the gather.
        idx = tl.load(i_ptr + r)
        tl.store(o_ptr + r, tl.load(x_ptr + idx))

    return ptr_gather_index_width_ods_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "i": torch.randperm(N, generator=g).to(torch.int16)}


def reference(inputs):
    # Never reached -- stage 3 refuses. Written out so the file still says what
    # the kernel MEANT, for whoever reads it the day the refusal moves.
    return {"o": inputs["x"][inputs["i"].long()]}


SPEC = KernelSpec(
    name="ptr_gather_index_width_ods_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i16", "o_ptr": "*fp32",
               "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("i", IN, "int16", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect="refuse",
    expect_stop=Stop("triton-compile", Reason.TRITON_PASS_FAILED),
    expect_why="tts.make_gather_scatter_tptr's ODS takes an i32 or i64 offset "
               "and nothing else, which is what lets index_buffer ship the "
               "index at its own width",
)
