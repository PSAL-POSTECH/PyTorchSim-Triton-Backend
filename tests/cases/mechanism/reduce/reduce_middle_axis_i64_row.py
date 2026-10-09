"""`tl.sum(x, axis=1)` on a rank-3 i64 tile -- the VECTOR ACCUMULATOR fold, sized.

WHAT SHAPE THIS IS. The reduced axis is neither the lanes nor the in-lane
vector: dim 0 is banked, dim 2 is what sits in a lane's vector register, and
dim 1 -- two elements -- is what folds. So each trip of the fold is one
whole-vector combine and the accumulator stays a vector the width of dim 2.
vectorize calls that PATTERN 2, and it is the one reduction shape whose
vector axis gets no loop of its own from the other three paths.

WHY i64 AND WHY 64. The width a vector may have is `VLEN * ISA_MAX_LMUL` bits
for ONE instruction -- 256 x 8 = 2048 here -- and 64 elements of i64 want 4096,
which is an LMUL of 16 the ISA cannot encode. f32 at the same extent is 2048 and
fits exactly, so the element type is the feature and not a dressing: this is the
smallest thing that asks the accumulator to be wider than one instruction.

WHAT BREAKS IT. Before the fix, `_bank_reduce`'s PATTERN 2 entered the vector
axis at 0 and covered it with the vector's own width -- no loop, so nothing
carried `torchsim.vector_loop` and legalize, which only ever narrows a marked
loop's step, never saw it. The width was final at stage 4 and the pipeline
refused it two stages later, with no file and no line between:

    a vector is wider than one instruction can reach: <64 x i64> needs 4096
    bits (LMUL 16). The budget is VLEN 256 x LMUL 8 = 2048 bits.

The fix is the one that path's own comment named: a one-trip loop over the
vector axis, marked, exactly as PATTERN 1 emits for the axis it folds. Then all
four reduction paths own no width and legalize owns all of them.

WHAT PROVES IT FIRED (08-transformed.mlir)
  Before, the fold stood on its own with the vector axis entered at 0:
      %3 = affine.vector_load %0[%arg8, 0] : ..., vector<64xi64>
      %4 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %3) -> (vector<64xi64>)
  After, it sits inside a marked loop that legalize sizes:
      affine.for %arg9 = 0 to 64 step 64 {
        %3 = affine.vector_load %0[%arg8, %arg9] : ..., vector<64xi64>
        %4 = affine.for %arg10 = 0 to 2 iter_args(%arg11 = %3) -> (vector<64xi64>)
      } {torchsim.vector_loop}
  64 at VLEN 512, where 64 i64 is 4096 bits -- LMUL 8, one instruction; at
  VLEN 256 the same loop is cut to 32. 11-kernel.s issues it at `e64, m8`.

WHERE IT WAS FOUND. Qwen2-MoE and Qwen3-MoE on transformers 5.15.0, whose
`grouped_mm_experts_forward` sorts the flattened (token, expert) pairs. Inductor
expands that `torch.sort` into a bitonic network, and one compare-and-swap step
is `tl.sum(iy * left_mask, 1)` over a [128, 2, 64] i64 reshape -- this shape,
with a whole transformer around it.

Vector-accumulator reduction (vectorize PATTERN 2) whose accumulator is wider than one instruction can issue. Pins that the vector axis of that path gets a marked loop, so legalize sizes it like every other vector axis. Integer sum, so the answer is exact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 128, 2, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_middle_axis_i64_row_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                          K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None, None] * (K * N)
                    + rk[None, :, None] * N + rn[None, None, :])
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.sum(x, axis=1))

    return reduce_middle_axis_i64_row_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-2 ** 40, 2 ** 40, (M, K, N), generator=g,
                               dtype=torch.int64)}


def reference(inputs):
    return {"o": inputs["x"].sum(dim=1)}


SPEC = KernelSpec(
    name="reduce_middle_axis_i64_row_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "o_ptr": "*i64", "M": "constexpr",
               "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "int64", (M, K, N)), Arg("o", OUT, "int64", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
