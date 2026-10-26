"""`tl.max(x, axis=1)` -- the same in-lane reduction as reduce_row_sum, and the
one that exposed the scalar register file.

WHY MAX AND NOT JUST SUM
The two differ in ONE thing at the ISA level: `vfredosum.vs` has an accumulator
operand (it folds vs1[0] in), `vfredmax.vs` does not. So when _bank_reduce
handed the outs init to `vector.reduction` as `acc`, LLVM kept the sum's combine
inside a vector register and spilled the max's out to an f-register:

    sum   vle32.v v10, (a1)          vfredosum.vs v8, v8, v10    vrgather.vi
    max   flw     fa5, 0(a1)         vfmv.f.s fa4, v8            fmax.s / vfmv.s.f

There is ONE scalar register file behind all 128 lanes. `vfmv.f.s` keeps lane 0's
value and discards the other 127, and `vfmv.s.f` then broadcasts that single
value back over every lane. Measured: err 2.86 on this kernel while the
identical row sum was exact -- a difference visible only in the assembly.

THE FIX
_bank_reduce never passes `acc`. The init is combined afterwards as an arith op
on vector<1>, which is the same combiner the linalg body named, so it stays
per-lane. Costs one vfmax.vv the sum case used to fold away.

verify_lane_values_stay_vector bans any op taking a vector to a scalar in
the LLVM IR so the next lowering that does this is a build failure, not a wrong
number. The INJECT direction (`vmv.s.x v8, zero`) stays allowed -- a constant is
the same in every lane, and relu already emits one.

WHAT PROVES IT FIRED
  11-kernel.s inner loop is vle32.v / vfredmax.vs / vrgather.vi / vfmax.vv /
  vse32.v, with no f-register anywhere.

The reduction whose combiner has no vector accumulator. Exact (err 0) because max is order-independent, so any drift is a lowering fault.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_row_max_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        tl.store(o_ptr + rm, tl.max(x, axis=1))

    return reduce_row_max_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].amax(dim=1)}


SPEC = KernelSpec(
    name="reduce_row_max_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
