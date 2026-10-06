"""A reduction LONGER than one vector register group -- split by NARROWING.

The scratchpad tile is what the Triton source asked for; how much of it one
instruction covers is this compiler's decision. A vector register group holds
VLEN_BITS * ISA_MAX_LMUL bits -- 64 f32 at vlen=256, LMUL 8 -- so a 128-wide
reduction does not fit.

THE SPLIT IS A RETYPE, NOT A RESHAPE
vectorize emits ONE accumulator loop over the whole in-lane extent, and
legalize narrows its step to what an instruction can issue. Measured,
04-passes/08 -> 09:

    %5 = affine.for %i = 0 to 128 step 128 iter_args(%acc = %4) -> vector<128xf32>
    %6 = affine.for %i = 0 to 128 step  64 iter_args(%acc = %5) -> vector<64xf32>
         %26 = affine.vector_load %1[%r, %i] : memref<128x128xf32, 1>, vector<64xf32>
         %27 = arith.addf %acc, %26 : vector<64xf32>

The BOUND never moves, so the same 128 elements are now covered in two trips and
no buffer appears -- there is nowhere for one to go. An earlier version reshaped
instead (tensor.expand_shape [128,128] -> [128,4,32], one linalg.reduce becoming
two, in a pass called split_reduction); it was removed in 653af58 once narrowing
did the same job without a second reduce to keep consistent.

TWO PATTERNS FALL OUT, AND THEY LOWER DIFFERENTLY
    the accumulator loop  each trip is one whole-vector combine, so the partial
                          results stay live in a vector register
                            affine.for iter_args(%acc) { vfadd.vv }
    the final fold        one element per lane, the single horizontal step
                            vector.reduction  ->  vfredosum.vs

Exactly one horizontal instruction no matter how long the axis is. Folding each
chunk horizontally would give the same answer and serialise every one of them.

NO IDENTITY TABLE
The partial accumulator needs the combiner's identity (0, 1, -inf, +inf,
all-ones...) and a wrong one is a wrong number rather than a compile error. It is
not picked: triton-shared already filled the reduction's own init with the right
one, so vectorize broadcasts that same scalar to the vector shape
(`%4 = vector.broadcast %3`).

    N =  32   2.86e-06     (fits in one vector, not split)
    N =  64   refused  ->  3.81e-06
    N = 128   refused  ->  5.72e-06

The error grows with N because it is fp32 accumulation order, not lowering drift.
reduce_row_max is the companion that must be exactly 0.

Reduced axis 4x the vector width. Covers the chunked accumulator in vectorize._bank_reduce and the compute tile, narrowed by legalize.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_reduce_multi_chunk_kernel(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        tl.store(o_ptr + rm, tl.sum(x, axis=1))

    return tile_reduce_multi_chunk_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    return {"o": inputs["x"].sum(dim=1)}


SPEC = KernelSpec(
    name="tile_reduce_multi_chunk_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M, N)), Arg("o", OUT, "float32", (M,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
