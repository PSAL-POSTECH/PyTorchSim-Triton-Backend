"""`s = tl.sum(x)` then `x - s` -- the rank-0 reduction result read back into a
tile, at its smallest.

WHAT IT PINS, AND WHY IT IS A PIN ON A DELETION
`fit_layout_to_lanes._replicate_captured_scalars` used to answer this shape. It
found the `tensor.extract` of a rank-0 reduce, saw the scalar CAPTURED in the
consuming `linalg.generic`'s region, and replaced it with a replicating
`torchsim.relayout` -- one element out to DRAM and back with a zero stride, so every
lane filled its own bank. Both halves it was avoiding are real: `tensor.extract`
puts a value in a scalar register and there is one scalar register file per
lane, and a captured scalar has to be broadcast into the vector, which is the
move between banks the hardware does not have.

It was deleted, and this kernel is the reason that is safe to do:

  the trigger is gone   it collects `tensor.extract`, and there are ZERO tensor
                        ops in fit_layout_to_lanes's input across all 256 kernels
                        -- bufferize turns the extract into a `memref.load` five
                        passes above. Measured: reached in 240 of 240 kernels,
                        matched nothing in any of them.
  the emitter is gone   a rank-0 source is the one shape `_emit_relayout` does
                        not lower; it raises "relayout over a rank-0 buffer".
                        So a trigger that fired again would kill the pass.
  the shape still works  which is what this file measures. 139 values are
                        captured from a `memref.load` inside a `linalg.generic`
                        body across 154 kernels of the suite, and the great
                        majority of those kernels pass.

WHAT WOULD BREAK IT. Reviving a rank-0 replicating relayout without an emitter
arm for it: the pass then dies here on its own marker rather than compiling.
Equally, a change that makes the captured `memref.load` reach the vector
lowering as a scalar shows up as `math.exp on a SCALAR has nowhere to go`, which
is what two other kernels in the suite fail on today -- a stage-5 problem, and
this kernel is where a stage-4 cause for it would appear first.

Deliberately rank-0, not `tl.sum(x, axis=1)`: reduce_row_sum.py already covers
the rank-1 fold, and it is the rank-0 one -- the whole tile to a single value,
then read by every element of the tile -- that has no in-lane answer at all.

A rank-0 reduction result captured in a linalg.generic body. Pins that the shape _replicate_captured_scalars was deleted for still compiles without it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 8, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_captured_scalar_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                      N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        idx = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + idx)
        s = tl.sum(x)                 # rank-0: one value for the whole tile
        tl.store(o_ptr + idx, x - s)  # captured in the consumer's body

    return reduce_captured_scalar_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x - x.sum()}


SPEC = KernelSpec(
    name="reduce_captured_scalar_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
