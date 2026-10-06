"""An `any` reduction: i1 in, i1 out, OR as the combiner.

Three things about a PREDICATE IN MEMORY meet in one op, and each of them was
broken on its own. No buffer holds i1 -- p04_bufferize._widen_i1_buffers gives a
predicate a byte per element, because the machine stores an i1 vector PACKED and
seven bytes in every eight would never be written -- so an i1 reduction is where
that widening has to survive contact with everything downstream.

    the accumulator's init   `linalg.fill` writing `false` into the i1 result.
                             Its value is a SCALAR OPERAND, not a block
                             argument, so the widening walked past it and left
                             an i1 value filling an i8 buffer:
                               'linalg.fill' op expected fill value type ('i1')
                                             to match output element type ('i8')
    the accumulator's read   the combiner reads the value it is writing. The
                             widening skipped `outs` arguments on the grounds
                             that they are "rarely read", and this body reads
                             one every iteration:
                               'arith.ori' op requires the same type for all
                                           operands and results  (i1, i8) -> i1
    the shape of the body    with both fixed, the body is trunci, trunci, ori,
                             extui -- and vectorize reads a reduce as
                             "elementwise steps, then a combiner over the
                             incoming element and the accumulator". The last op
                             is now a CAST and one of the steps in front of it
                             reads the ACCUMULATOR, which _bank_reduce excludes
                             on purpose because that is what keeps the fold
                             associative:
                               NotVectorisable: reduce body does not end in a
                               known combiner

The third is answered where it is made rather than where it is noticed: `and`,
`or` and `xor` mean the same thing at any width on a zero-extended value, so the
trunc-compute-extend round trip is exactly the identity and the body is rebuilt
one byte wide (_drop_cancelling_casts). vectorize's rule is about
associativity and is left alone.

    measured   tests/ops/attention/test_gqa.py through the Triton route, whose
               `_safe_softmax` kernel calls `triton_helpers.any(mask, 1)`. It is
               in PyTorchSim's gating allowlist and had stopped at the first of
               the three; with all three it passes at 1.42e-07. Each failure
               above was uncovered by fixing the one before it, so the three are
               one kernel and not three.

WHY OR AND NOT A COMPARISON. `tl.max` of a bool would reduce in i32 after
promotion and never build an i1 accumulator; `tl.reduce` with an explicit `|`
keeps the whole fold in the predicate type, which is the case the widening is
about.

torch equivalent: (x > 0).any(dim=1), as float.

an OR fold over an i1 tile: the fill value, the accumulator read and the cast shell the i1-to-i8 widening leaves behind.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM = 32
BN = 16
NX = BM * BN


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _or_combine(a, b):
        return a | b

    @triton.jit
    def reduce_i1_any_or_kernel(x_ptr, o_ptr, BM: tl.constexpr,
                                BN: tl.constexpr):
        om = tl.arange(0, BM)
        on = tl.arange(0, BN)
        v = tl.load(x_ptr + om[:, None] * BN + on[None, :])
        m = v > 0.0                                  # an i1 TILE
        anym = tl.reduce(m, 1, _or_combine)          # i1 accumulator, i1 result
        tl.store(o_ptr + om, tl.where(anym, 1.0, 0.0))

    return reduce_i1_any_or_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(NX, generator=g, dtype=torch.float32)
    # Rows that are ALL negative and rows that are not, both present: a fold
    # that came home constant would still match if every row answered true.
    x = x.reshape(BM, BN)
    x[::4] = -x[::4].abs() - 1.0
    return {"x": x.reshape(NX).contiguous()}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(BM, BN)
    return {"o": (x > 0.0).any(dim=1).to(torch.float32)}


SPEC = KernelSpec(
    name="reduce_i1_any_or_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BM": "constexpr",
               "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (BM,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
