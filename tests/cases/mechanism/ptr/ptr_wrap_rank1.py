"""A rank-1 wrap: the axis that used to lose its modulo silently.

`w1_ptr + (rows % WM)` with rows starting at `pid * BLOCK`. PtrAnalysis records
the modulus in `shape: [WM]` exactly as it does at rank 2, and there is no
second axis to split against.

THIS IS A CORRECTNESS FIX THAT ALREADY LANDED, kept as its pin.
StructuredToMemref looked at the rank, saw 1, and sent the pointer down the
plain reinterpret_cast path (`op.isSplitPtr() && getStaticShape().size() == 1`),
DROPPING the wrap it had been handed. That pass is deleted (rule 21) and this
backend never ran it. b11 sees a wrapped axis it cannot cut into runs
(the start is runtime, so _wrap_cut cannot count the laps) and emits an
indirect transfer over the address tile, where `mod W` is what the index holds.
The wrap is honoured, and the numeric check below is what says so.

COVERS (1)   a02_convert_tts_to_pytorchsim.py, _emit_per_element at rank 1

WHY IT IS NOT OUR BUG
That pass is DELETED (rule 21), and was never in this pipeline anyway. Stage 2
stops at triton-ptr-to-memref with
the tts ops still standing (see lowering/triton_compile.py), and
convert_accesses_to_transfer reads `shape:` itself -- a non-zero entry becomes an
indirect transfer whose index tile carries the modulo, at any rank. So the
branch that drops it is never reached, and the numbers come out right:

    got == w1[arange(32) % 24]        exact
    got != w1[arange(32)]             which is what the dropped form would give

reference() used to be zeros(), with a note that "predicting the emitted (wrong)
values is not something a torch reference should endorse". That was true of the
path it was describing. Here the kernel computes what it says it computes, so
the reference is the wrap.

THE BUFFER IS STILL OVERSIZED
w1 has 64 elements although WM is 24, from when the generated code really did
read w1[0:32]. It is harmless now -- the extra elements are never addressed --
and it stays so the file keeps documenting the shape of the upstream hazard.

A rank-1 modulo whose wrap upstream used to drop at rank 1. convert_accesses_to_transfer reads `shape:` itself, so the wrap is applied and the numbers are exact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

WM = 24         # the wrap extent PtrAnalysis records in `shape:`
BLOCK = 32
NW = 64         # oversized on purpose, see the docstring


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_wrap_rank1_kernel(w1_ptr, o_ptr, WM: tl.constexpr,
                                      BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        rows = pid * BLOCK + r
        v = tl.load(w1_ptr + (rows % WM))      # rank-1 wrap: never split
        tl.store(o_ptr + r, v)

    return ptr_wrap_rank1_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w1": torch.randn(NW, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(BLOCK)
    return {"o": inputs["w1"][r % WM]}


SPEC = KernelSpec(
    name="ptr_wrap_rank1_kernel", kernel=kernel,
    signature={"w1_ptr": "*fp32", "o_ptr": "*fp32", "WM": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"WM": WM, "BLOCK": BLOCK},
    args=[Arg("w1", IN, "float32", (NW,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
