"""A short-K matmul whose A operand is +inf in the lanes the tile does not
occupy -- the array must not multiply them.

WHAT THIS PINS. K = 16 of a 128-lane array, and on the INPUT push side the
array's k axis IS THE LANE AXIS. So lanes 16..127 of the A tile are outside the
tile, and lower_to_vcix has to push zero for them. It already pushes zero for
the matching WEIGHTS -- `if i < K: ... else: zero_vector`, which
mixed_gemm_partial_k covers -- and until this kernel it pushed A across all 128
lanes unmasked. The array then computed

    0 (weight past K)  *  inf (input past K)  =  0x7fc00000

in EVERY output element, not just the lanes it happened in: an IEEE invalid
operation poisons the accumulator, and the accumulator is shared.

WHY THE +inf HAS TO BE ASKED FOR, and why a row-sum divide is the smallest way
to ask. A dead lane holds whatever the last banked loop left in that bank, and
vectorize emits those loops over ALL 128 lanes. The row sum arrives as a
stride-0 MVIN that writes only the 16 live lanes, so its dead lanes are the
untouched zeros of a fresh scratchpad allocation -- and x/0 is +inf there while
the 16 live lanes do an ordinary normalisation with nothing unusual about it.
That is the shape every kernel that hit this in the wild had:
dl/attention_noloop, dl/flux_single_block, dl/chunked_prefill_attn and
dl/rel_shift_attn all fed `exp(s-m)/sum(exp(s-m))` to a dot and all returned
nan. This is that sentence with the attention taken out.

    before  max_abs_err = nan   -- all 1024 output words are 0x7fc00000, bit
                                  for bit, on the tree at 53d9318
    after   max_abs_err = 1.19209e-07

THE DIVISOR COMES FROM A SECOND TENSOR, and that is not decoration. Summing x
itself and then streaming x into the matmul asks fit_layout_to_lanes for two lane
axes on one buffer and it refuses ("no axis every matmul can stream"), which is
a different, open defect -- measured across M in {32,64} x K in {16,32} x N in
{16,32,64}, all twelve refused, with or without a transposed B. Reading the
divisor off `y` keeps this kernel about the push and not about that refusal.

WHAT BREAKS THIS. Removing the lane mask on the A push (an arith.select on
pytorchsim.lane_id < K) puts the nan straight back. So does gating it on anything
but `K < lanes`: a matmul that fills the array must emit no mask, and one that
leaves lanes over must emit one. An epsilon on the divisor also makes the
numbers come out -- which is why the guard is placed on the pushed vector and
not on the arithmetic that happens to produce the inf.

SIZES. M = 32 so the M push loop runs more than once. K = 16 is the smallest
`tl.dot` accepts and so leaves the most lanes dead (112 of 128). N = 32. None of
the three is 128, so a fix that only works on the square case cannot pass here.

a matmul with K below the lane count whose A operand is +inf in the dead lanes -- the input push must mask them to zero, the way the weight push already does.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 32
K = 16          # 16 of 128 lanes -- 112 dead ones on the A push
N = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def dead_lane_inputs_kernel(x_ptr, y_ptr, w_ptr, o_ptr, M: tl.constexpr,
                                K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])       # [M, K]
        y = tl.load(y_ptr + rm[:, None] * K + rk[None, :])       # [M, K]
        w = tl.load(w_ptr + rk[:, None] * N + rn[None, :])       # [K, N]
        # Finite in all 16 live lanes; the divisor is 0 in the 112 dead ones,
        # so `a` is +inf there and the array must not see it.
        a = x / tl.sum(y, axis=1)[:, None]                       # [M, K]
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, w))

    return dead_lane_inputs_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # STRICTLY POSITIVE, so the row sum is never near zero: a 1/0 the KERNEL
    # asks for would be a different kernel, and this one is about the 1/0 the
    # backend asks for in lanes the kernel never mentions.
    return {"x": torch.rand(M * K, generator=g, dtype=torch.float32) + 0.5,
            "y": torch.rand(M * K, generator=g, dtype=torch.float32) + 0.5,
            "w": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, K)
    y = inputs["y"].reshape(M, K)
    w = inputs["w"].reshape(K, N)
    return {"o": ((x / y.sum(dim=1, keepdim=True)) @ w).reshape(-1)}


SPEC = KernelSpec(
    name="dead_lane_inputs_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "w_ptr": "*fp32",
               "o_ptr": "*fp32", "M": "constexpr", "K": "constexpr",
               "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M * K,)), Arg("y", IN, "float32", (M * K,)),
          Arg("w", IN, "float32", (K * N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
