"""`tl.arange` used as a VALUE, not folded into an address.

Normally PtrAnalysis absorbs `tl.arange` into the pointer arithmetic and the
range never becomes a tensor. It survives when the value itself is consumed --
as a gather offset, say (`x_ptr + (r % M)`), which is how ptr_gather_modulo and
ptr_gather_1d_arith reach it. triton-shared then hands us

    linalg.generic { linalg.index(0) -> index_cast -> yield }

with no inputs at all, and vectorize used to refuse it: `linalg.index` is
not one of the arith/math ops it knows how to widen.

WHAT MAKES IT WORK

A dim0-on-lanes tile puts element i in LANE i, so a dim-0 `linalg.index` IS the
lane number, and the hardware has an instruction that reads it -- Spike's
torchsim_vlane_idx writes vu_idx into every element of that lane's vector.
vectorize emits `torchsim.vlane_idx` for it and lower_to_vcix turns that into
the VCIX op, the same instruction PyTorchSim reaches through its own
torchsim.vlane_idx (which multiplies by a stride; we want the raw index, so
there is no multiply).

The placeholder stays unregistered for the same reason torchsim.transfer does:
stage 3 writes text and stage 4 re-parses it, and `torchsim.*` is ours to name.
The `vcix.*` ops DO round-trip -- the LLVM this compiler pins registers them.

WHY THIS KERNEL LOOKS ODD
  `i + r` rather than just `r`: the load is what stops PtrAnalysis from folding
  the range away, and the add is what forces the range to be a value. Storing a
  bare `tl.arange` would be constant-folded before any of this.

torch equivalent: o = x + arange(BLOCK), with x all zeros -> o == arange(BLOCK).
That is the point: a wrong lane index shows up as a wrong element, in order.

linalg.index -> lane number via torchsim.vlane_idx. The only direct check that the lane id is the element index.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_arange_as_value_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        i = tl.load(x_ptr + r)
        tl.store(o_ptr + r, i + r)      # r as a VALUE -> linalg.index survives

    return ops_arange_as_value_kernel


def make_inputs(torch, seed=0):
    return {"x": torch.zeros(BLOCK, dtype=torch.int32)}


def reference(inputs):
    import torch
    return {"o": torch.arange(BLOCK, dtype=torch.int32)}


SPEC = KernelSpec(
    name="ops_arange_as_value_kernel", kernel=kernel,
    signature={"x_ptr": "*i32", "o_ptr": "*i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "int32", (BLOCK,)), Arg("o", OUT, "int32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
