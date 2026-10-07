"""GEMM with fp16 operands and an fp32 accumulator -- the only MIXED matmul here.

`tl.dot` on half inputs does not produce a half matmul. It produces a genuinely
MIXED linalg op,

    linalg.matmul ins(%A, %B : memref<...xf16>, memref<...xf16>)
                  outs(%C : memref<...xf32>)

and that is the whole point of this file: every other matmul in this suite has
ONE element type, so a pass that reads the element off A and uses it for C as
well cannot be told apart from one that reads both. pytorchsim_triton_compiler/lowering/passes/p08_lower_matmul.py
did exactly that until 3dcde65 -- one `elt` typed the pushes, the compute op's
sew, the pop, the C read, the accumulate and the write. On an all-f32 gemm those
are the same type and the bug is invisible.

WHAT IT DEFENDS, and what breaking each half looks like

  TWO ELEMENT TYPES, NOT ONE. `elt` (f16) types the A/B pushes and the sew the
  array multiplies at; `acc_elt` (f32) types the pop, the C transfer_read, the
  arith.addf and the transfer_write. Collapse them back into one and the module
  stops verifying long before it is a wrong number -- measured against the tree
  before 3dcde65, stage 4:

      'vector.transfer_read' op requires formal padding and source of the
      same elemental type
        ... (memref<128x128xf32, 1>, index, index, f16) -> vector<16xf16>

  which is the f32 C tile being read with A's element and A's zero pad.

  ONE REGISTER'S WORTH, PER SIDE. A vector register is `vlen` BITS, so the
  element count is a property of the element. At vlen=256 this kernel pushes
  8 x 16 halves and pops 16 x 8 floats. Deriving the pop count from `elt` gives
  vector<16xf32> -- two registers -- and silently doubles the LMUL of every pop.
  Nothing rejects that; it just reads the wrong half of the next register.

  THE INTEGER TRANSPORT. The VCIX push/pop is a raw register handoff, and stock
  LLVM only instantiates the XSfvcp patterns over types a NAMED extension makes
  legal. The target here is +v,+zfh, and `zfh` is the SCALAR half extension --
  it says nothing about vector element types, so <16 x half> is not legal. The
  payload therefore travels as the same-width INTEGER vector with an
  arith.bitcast at the boundary (RVV has an integer type for every SEW, e16/m1
  is the same vtype either way, and a bitcast moves no bits). Type the VCIX
  operands with the float again and this kernel dies in stage 5, inside llc's
  type legalizer:

      LLVM ERROR: Do not know how to split this operator's operand!
        Running pass 'RISC-V DAG->DAG Pattern Instruction Selection'

  which names nothing whatsoever about halves. fp32 never reaches that message,
  because <8 x float> IS legal under plain +v -- so this kernel is the only
  thing in the suite standing between that transport and a revert. (Removing
  only the bitcast and leaving the word typing fails earlier and differently,
  `arith.addf ... (vector<8xf32>, vector<8xi32>)` at stage 4: the two are one
  mechanism and neither half survives on its own.)

WHY THIS SHAPE AND NOT 256 CUBED
  A 256^3 fp16 gemm gives max_abs_err exactly 0, and the zero is real but
  useless as a witness: an fp16 product has at most 22 significant bits, an f32
  mantissa holds 24, so every product is EXACT in the accumulator and the only
  rounding left is the sum -- which at that shape happens to match torch's
  order. An error column that reads 0 cannot distinguish "accumulated in fp32"
  from "accumulated in something else that also came out even", and a kernel
  whose headline number is a zero teaches the next reader nothing about which
  digits are load-bearing. 256 x 128 x 384 orders the sum differently from
  torch's and answers 5.72205e-05 -- small enough to prove the accumulator is
  f32 (an f16 accumulator over K=384 would be orders of magnitude coarser, not
  five decimal places) and non-trivial enough to move if anything under it does.

  K = 3 * BK also keeps the reduction loop alive: one K tile and triton unrolls
  it away, and lower_to_vcix would never see the loop-carried accumulator that
  makes this a reduction rather than three separate matmuls.

WHAT WOULD BREAK IT
  Reading one element type off the operands again. Sizing the pop from the push
  element. Removing the bitcast at the VCIX boundary, or narrowing it to "if the
  element is f16" -- it is applied to every type on purpose, and the fp32 gemm's
  generated assembly is byte-identical with it, so there is no fp32 kernel that
  would notice its removal. And anything that widens the operands to f32 before
  the matmul: the numbers would still be right, and the mixed op this file
  exists to carry would be gone.

fp16 operands, fp32 accumulator: the only matmul in the suite whose linalg op carries two element types. Pins lower_to_vcix's elt/acc_elt split, the per-side register capacity, and the integer transport with a bitcast at the VCIX boundary.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Dim 0 of every tile divides by TORCHSIM_COMPILE_VECTORLANE_SIZE (128) -- the DMA splits it
# across lanes. N and K are at the systolic size; M is two tiles and K is three,
# so the grid is 2 x 1 and the reduction loop runs three trips.
M, N, K = 256, 128, 384
BM, BN, BK = 128, 128, 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_fp16_kernel(a_ptr, b_ptr, c_ptr,
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        # f32 accumulator on f16 loads -- this is what makes the linalg.matmul
        # mixed. `tl.dot` takes its output type from `acc`, and the operands
        # keep theirs: ins(f16, f16) outs(f32).
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return gemm_fp16_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(M, K, generator=g, dtype=torch.float32).to(torch.float16),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.float16),
    }


def reference(inputs):
    import torch
    # Widened to f32 BEFORE the matmul, deliberately: that is what the hardware
    # does -- the array multiplies in the input format and accumulates in float
    # (spike's systolic_array.h carries the accumulator as C `float`). A torch
    # matmul left in fp16 would accumulate in half and be the wrong reference to
    # compare an f32 accumulator against.
    return {"c": torch.matmul(inputs["a"].to(torch.float32),
                              inputs["b"].to(torch.float32))}


SPEC = KernelSpec(
    name="gemm_fp16_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK},
    args=[
        Arg("a", IN, "float16", (M, K)),
        Arg("b", IN, "float16", (K, N)),
        Arg("c", OUT, "float32", (M, N)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    make_inputs=make_inputs,
    # The inputs are exact in f32 and the accumulator is f32, so what is left is
    # summation ORDER -- the array's K walk against torch's. Measured
    # 5.72205e-05 absolute, 0.0202492 relative: the relative figure is large
    # because some entries of C are near zero, which is why the pass is decided
    # by atol at this scale and not by rtol.
    rtol=1e-3, atol=1e-3,
)
