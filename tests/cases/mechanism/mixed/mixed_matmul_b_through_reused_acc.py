"""A RED PIN. A matmul whose accumulator buffer is REUSED as an elementwise
operand of its own A operand gets its B pushed TRANSPOSED, silently.

    x = tl.load(...)   [32, 32]        w = tl.load(...)   [32, 32]
    tl.dot(x, w)                    -> green
    tl.dot(x + x, w)                -> green
    tl.dot(x * 2.0, w)              -> max_abs_err 62.6708, EVERY element wrong

The result is exactly `(x * 2.0) @ w.T` -- measured, 1.14e-05 against that and
62.67 against the reference. Nothing refuses it and nothing warns.

WHAT MAKES THE THIRD LINE DIFFERENT, and it is not the arithmetic. `x * 2.0`
materialises the 2.0 splat as its own [32, 32] tile, bufferize hands that
allocation back out as the matmul's accumulator, and the SAME buffer is now
both an elementwise operand sharing A's iteration space and the matmul's C.
Through the generic its axis 1 is A's k; through the matmul its axis 1 is C's
n. The axis graph therefore merges k and n into ONE class, which is correct --
one physical axis of one allocation cannot be banked two ways.

WHERE THAT TURNS INTO A WRONG NUMBER. B[K, N] is the one buffer that has BOTH
k and n, so after the merge both of its axes sit in one class.
`LaneAxes.axis_of` answers "which axis of this buffer is in class c" out of a
dict built with `setdefault`, which keeps the LOWER axis -- so B comes back
banked on 0, its k, and the matmul is stamped

    pytorchsim.lane_axis = array<i64: 1, 0, 1>

against the MUST that put it on 1 ("linalg.matmul holds B, a column per lane").
lower_to_vcix never reads `pytorchsim.lane_axis` at all -- zero occurrences in the
file -- so it pushes B as if it were banked on N whatever the stamp says, and a
B banked on K is a B pushed transposed.

WHAT THIS IS NOT. Not the fill defect that mixed_matmul_a_written_in_sram pins:
that one is the 2.0 splat's own fill covering half its bank, it only bites when
the tile is not square, and fixing it leaves this kernel's 62.6708 UNCHANGED to
the last digit. The two are separate and this one is the reason that kernel's
K = 32 form is wrong in every row rather than in half of them.

WHAT ELSE HOLDS THE SAME SHAPE. dl/rel_shift_attn (`o` off by 0.136647),
dl/varlen_packed_attn and mixed/mixed_matmul_output_as_operand. dl/stft_frame
carries the same [1, 0, 1] stamp and PASSES -- its B is a DFT twiddle matrix,
symmetric to within 3.05e-05, so a transposed push is invisible in it. Its
recorded errors (4.96e-05, 5.72e-05, 4.10e-05) are that asymmetry and nothing
else; they drop to 1.9e-06 the moment B is banked right.

RED: reusing the matmul's accumulator as an elementwise operand of its A merges k with n, axis_of then hands B the lower of its two axes, and lower_to_vcix pushes the weights transposed. Result is exactly A @ Bt. Not the fill defect -- that fix leaves this at 62.6708.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 32, 32, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def b_through_reused_acc_kernel(x_ptr, w_ptr, o_ptr, M: tl.constexpr,
                                    K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])       # [M, K]
        w = tl.load(w_ptr + rk[:, None] * N + rn[None, :])       # [K, N]
        # The splat is the whole kernel: it mints the buffer that bufferize
        # then reuses as the accumulator. `x + x` computes the same A and goes
        # green, because it mints nothing.
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(x * 2.0, w))

    return b_through_reused_acc_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K, generator=g, dtype=torch.float32),
            "w": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, K)
    w = inputs["w"].reshape(K, N)
    return {"o": ((x * 2.0) @ w).reshape(-1)}


SPEC = KernelSpec(
    name="b_through_reused_acc_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M * K,)), Arg("w", IN, "float32", (K * N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
