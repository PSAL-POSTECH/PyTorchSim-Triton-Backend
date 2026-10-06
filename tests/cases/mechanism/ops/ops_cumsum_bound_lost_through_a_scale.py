"""An i32 wrapped by 2^32 before a cumsum -- a bound that arithmetic destroys.

Written to pin truncate_accumulation's refusal: that pass walked ONE op back
from the accumulation to find what had been extended, and would only walk
through a BROADCAST, because any other generic can scale the bound away. This
kernel is the scale. `m.to(int64) * K` has the extension as its first operand
and says nothing about how large it makes the result, so the bound the pass
would have read is not a bound at all.

THE PASS IS GONE and this kernel is kept anyway. It costs nothing -- the answer
is exact and always was -- and the shape it holds is the one that makes reading
a bound off an op unsound. Whatever narrows an accumulation next has to decline
this same IR, and it will be here to say so.

An arithmetic generic where a bound-reading pass expects a broadcast. Pins that the bound is only carried through an op that carries VALUES through.
"""

#: WHERE THIS STOPS NOW, TRACED 2026-09-23. It used to refuse in adapt: a
#: subview takes [255][512] of a 768-long tile, which picks a bank and not an
#: address. The slice arm answers that with one bank, and the kernel then reaches
#: spike and dies there -- `VI_ST_WHOLE ERROR: Accessing invalid spad address`.
#:
#: A refusal moved to a crash, which is the wrong direction (philosophy 12). The
#: crash is in the ONE_LANE emission, the same family `ops_reduce_one_lane`
#: pins: 768 i64 in one bank is more than a bank holds, and nothing on the way
#: down says so. The rung that knows the bank is the one that has to refuse.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

C = 512
K = 1 << 23      # 512 * 2^23 = 2^32 -- one full i32 wrap


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_cumsum_bound_lost_through_a_scale_kernel(
            m_ptr, o_ptr, C: tl.constexpr, K: tl.constexpr):
        c = tl.arange(0, C)
        m = tl.load(m_ptr + c) != 0
        # ONE generic between the extension and the accumulation, and it is not
        # a broadcast.
        v = m.to(tl.int64) * K
        o = tl.cumsum(v, axis=0)
        tl.store(o_ptr + c, o)

    return ops_cumsum_bound_lost_through_a_scale_kernel


def make_inputs(torch, seed=0):
    return {"m": torch.ones(C, dtype=torch.int32)}


def reference(inputs):
    import torch
    v = (inputs["m"] != 0).to(torch.int64) * K
    return {"o": torch.cumsum(v, dim=0)}


SPEC = KernelSpec(
    name="ops_cumsum_bound_lost_through_a_scale_kernel", kernel=kernel,
    signature={"m_ptr": "*i32", "o_ptr": "*i64", "C": "constexpr",
               "K": "constexpr"},
    constexprs={"C": C, "K": K},
    args=[Arg("m", IN, "int32", (C,)), Arg("o", OUT, "int64", (C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
