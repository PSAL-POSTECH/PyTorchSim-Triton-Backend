"""A REVERSE prefix scan, which upstream used to drop on the floor.

    out[i] = in[i] (+) in[i+1] (+) ... (+) in[n-1]

WHAT THIS IS THE WITNESS FOR

  `tl.cumsum(x, reverse=True)` lowers to a tt.scan with `reverse = true`, and
  the linalg lowering's CumSumConverter (ConversionPatterns.hpp:2113) read getAxis()
  and never getReverse(). ttx::CumSumOp has nowhere to put the flag -- input,
  axis, destination, and nothing else -- so the reverse scan became a FORWARD
  one with no diagnostic anywhere. Not a refusal, not a remark: a wrong answer.

  It is not even refusable from this side, because by the time the op reaches
  us the flag is gone. triton_shared 5ec6249 (`CumSumConverter: do not swallow
  a reverse scan`) fixes that at
  the source: a reverse scan now fails isCumSum()'s gate and stays a tt.scan,
  region and flag both intact.

  So this file is a witness for the PATCH as much as for the lowering. If the
  patch is ever lost -- a re-clone, a rebase -- the arms below stop matching
  their reference and say so, which is the whole reason it exists as a kernel
  and not as a comment.

HOW IT IS LOWERED

  The same Hillis-Steele the forward scan uses, pointed the other way: the pad
  goes at the BACK, the neighbour is read at +k instead of -k, and a slot takes
  the combined value where it HAS a neighbour (i < n - k rather than i >= k).
  Upstream spells the same idea flip(scan(flip())) (ScanOpToLLVM.cpp:492),
  which also settles the argument order -- the accumulated side is the first
  operand in both directions, so the combiner is untouched.

  Both arms are here because they take different paths to it: cumsum's region
  is a lone addf, which is exactly what isCumSum() looks for and now declines
  on the flag; cumprod's is a mulf, which never matched in the first place.
  They must agree on the direction regardless.

torch equivalent: x.flip(0).cumsum(0).flip(0), and the same for cumprod.

WHY THIS IS ALSO THE CROSS-LANE ROTATION'S WITNESS

  A rung reads a slice of the padded buffer that starts part-way along the axis
  it is banked on, and a lane cannot read another lane's bank. The crossbar's
  `permute` answers exactly that: it hands every lane the row its neighbour
  held, leaving the depth index alone, so the slice starts at zero again.

  THE DIRECTION IS WHY IT IS THIS FILE AND NOT THE FORWARD SCAN. A forward
  ladder's home is the pad, so it also WRITES through a shifted slice, and a
  rotation the other way would write every lane -- the ones outside the slice
  too. A slice is a masked store and the crossbar has no mask. A reverse
  ladder's home is zero: only its READS are shifted, and a read is a whole new
  destination, so no mask is wanted. convert_unsupported_ops stamps `torchsim.no_lane_axis` for the
  forward direction alone for that reason, and this file is what holds it there.

  TWO ARMS ARE WHAT MAKES IT A TEST OF THE SNAPSHOT. bufferize gives both
  ladders the same padded buffer and CSE gives both one slice op per offset, so
  a rotation minted once at the slice would be read by the second arm AFTER the
  first arm overwrote the source -- ONE IN FRONT OF EACH READER is the claim,
  and there are twelve readers. Measured with half of them the sum arm is right
  to 1e-7 and the product arm is wrong by 179x, which no count would have said.

Witness for triton_shared 5ec6249 (CumSumConverter reverse): without it the reverse flag is dropped and both arms answer the forward scan. Also the cross-lane rotation's witness -- a reverse ladder shifts its reads and its writes, and a permute answers both.
"""

#: RE-PINNED TWICE ON 2026-09-23. The old pin held 36 cross-lane rotations. The
#: reverse ladder reads `x[0:N-1]` against `x[1:N]`, and a slice that starts at
#: 1 along the lane axis picks a BANK. One bank was the answer for an hour and
#: cost the tile all 64 of its lanes; now each use of such a slice gets a buffer
#: of its own and the ladder runs spread, paid for in 24 trip legs. Numbers
#: exact (9.5e-07, TORCHSIM_COMPILE_IGNORE_EXPECT_IR=1).
#:
#: The first witness (triton_shared 5ec6249, the reverse flag) is unaffected --
#: it lives in the reference comparison. THE CROSS-LANE HALF MOVED OUT, and
#: `ops_xlu_all_gather_i64` and `ops_xlu_all_gather_tile` still pin the crossing
#: XLU op. IT IS ALSO WHERE THE 24 LEGS GO BACK: one crossing pass per shift is
#: what this kernel used to take, and that is the XLU round.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_scan_reverse_kernel(x_ptr, o_ptr, p_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        a = tl.load(x_ptr + r)
        tl.store(o_ptr + r, tl.cumsum(a, axis=0, reverse=True))
        tl.store(p_ptr + r, tl.cumprod(a * 0.01 + 1.0, axis=0, reverse=True))

    return ops_scan_reverse_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    return {"o": x.flip(0).cumsum(0).flip(0),
            "p": (x * 0.01 + 1.0).flip(0).cumprod(0).flip(0)}


SPEC = KernelSpec(
    name="ops_scan_reverse_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "p_ptr": "*fp32",
               "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,)),
          Arg("p", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # NO ROTATION, AND THE SHIFT GOES ROUND THROUGH DRAM. Each shifted read is
    # a slice that starts part way along the lane axis, and each use of one gets
    # a buffer of its own -- twelve of them, two legs each.
    expect_ir={"08-transformed.mlir": {"torchsim.crosslane = 16 : i64": 24,
                                   "dram_arg = -1": 0}},
)
