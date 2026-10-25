"""`tl.exp` of a RANK-0 REDUCTION -- a transcendental with no tile under it.

WHAT IT PINS. There is no scalar exp on this machine. The seven transcendentals
the vector unit has (exp, erf, tanh, sin, cos, log, atan) are `sf.vc.v.iv`,
which takes a VECTOR register, and the scalar float unit has no form of them at
all -- so a `math.exp` on an f32 reaches lower_to_vcix with nothing to push and
the LLVM tail behind it emits `call expf` into a libm that is not linked.

`tl.max(x)` with no axis folds the whole tile to ONE value. The linalg lowering writes
that as `linalg.reduce -> tensor<f32>` followed by `tensor.extract %r[]`, and
everything after the extract is scalar -- so `tl.exp` of it is the scalar shape,
at its smallest. scalar_math_to_tile is what answers it: the op moves onto a
one-element tile, and vectorize/legalize issue that tile as a vector
the way they issue every other one.

WITHOUT THAT PASS this dies in stage `lower` with

    math.exp on a SCALAR has nowhere to go: VCIX takes a vector register

which is the error two of the suite's dl kernels (flash_decode_splitk,
gated_delta_rule) used to carry. Verified against the tree before the pass.

WHY THE REDUCTION TAIL DOES NOT ALREADY COVER IT. legalize floors its
fold at two elements on purpose (_fold_tree/_spread_pair) so a reduction never
ends at a vector<1> that llc could scalarise. That floor is real and it is
reached here -- and it does not help, because the scalar is not made by the
fold. It is made by the `tensor.extract` AFTER the fold, which is the program
asking for one value, and no width of fold removes it.

THE TILE IS 1-D ON PURPOSE. A rank-0 `tl.max` over a 2-D tile is one
`linalg.reduce` over BOTH dimensions, and whether that survives as far as
vectorize -- which refuses two axes at once -- depends on fusion several
passes above. reduce_rank0_two_axis.py holds that shape; measuring it again here
would mean this file's failures are sometimes about something else.

DELIBERATELY NOT `tl.max(x, axis=...)` ON A 2-D TILE: an axis reduction leaves a
tile, the exp lands on that tile, and the kernel passes with or without the fix
-- which would pin nothing (rule 2). It is the WHOLE-tile fold that has no tile
left.

WHAT WOULD BREAK IT. Moving the rewrite below bufferize (the value is a
`memref.load` by then, and the way back out of a vector register is a vector
made scalar, which c27_verify_lane_values_stay_vector bans). Or teaching lower_to_vcix a scalar
arm instead: that is the libm call this refuses.

A whole-tile reduction feeding tl.exp. The scalar has no vector register, and this machine has no scalar transcendental.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_scalar_exp_kernel(x_ptr, o_ptr, N: tl.constexpr):
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rn)
        m = tl.max(x)                  # rank-0: no tile left
        w = tl.exp(0.0 - m)            # the transcendental, on a SCALAR
        tl.store(o_ptr + rn, x * w)

    return reduce_scalar_exp_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    return {"o": x * torch.exp(-x.max())}


SPEC = KernelSpec(
    name="reduce_scalar_exp_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
