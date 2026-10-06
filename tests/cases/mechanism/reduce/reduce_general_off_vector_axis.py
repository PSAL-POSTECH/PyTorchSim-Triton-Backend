"""A MULTI-RESULT reduce over an axis that is not the in-lane vector axis.

`tl.argmax` arrives as a two-result `linalg.reduce`, which sends it to
vectorize._bank_reduce_general -- and that path refuses `rdim != vdim`,
while the single-combiner path right above it HANDLES exactly that case (its
`folds_vector = False` branch, taken 60 times in this suite). One combiner and
it works; two and it does not. That is a hole, not a duplication.

WHY RANK 3. At rank 2 the shape cannot reach the refusal: the two axes are the
lane axis and the in-lane vector axis, and folding the lane axis is refused one
check earlier ("would have to fold across lanes"). With three axes a tile banked
on axis 0 has its vector on axis 2, so folding axis 1 is neither.

WHAT WOULD BREAK THIS
  * making the reduce single-combiner (tl.max instead of tl.argmax) -- it then
    takes _bank_reduce and passes, which is the point of the file
  * reducing axis 2 instead -- rdim == vdim and the general path handles it
  * a rank-2 tile -- unreachable, see above

Pins the _bank_reduce_general hole: rdim != vdim is refused there while _bank_reduce's folds_vector=False branch handles it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 8, 16, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_general_off_vector_axis_kernel(
            x_ptr, o_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        off = rm[:, None, None] * (K * N) + rk[None, :, None] * N + rn[None, None, :]
        x = tl.load(x_ptr + off)
        am = tl.argmax(x, axis=1)
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], am)

    return reduce_general_off_vector_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, K, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    return {"o": inputs["x"].argmax(dim=1).to(torch.int32)}


SPEC = KernelSpec(
    name="reduce_general_off_vector_axis_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*i32",
               "M": "constexpr", "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M, K, N)),
          Arg("o", OUT, "int32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
