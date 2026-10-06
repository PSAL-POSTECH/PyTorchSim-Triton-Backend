"""`% M` on an index -- the three shapes visitOperandRem accepts, and what the
modulus does to the `shape:` (wrap-around) list of tts.make_tptr.

Split out of the old tests/cases/ptr_structured.py section B. A modulo is
the only way a Triton address can be non-monotonic and still be structured:
PtrAnalysis records the modulus in PtrState::shape instead of emitting an
arith.remsi, and p06/p16 later turn a non-zero shape entry into one DMA piece
per lap. This file is about the FIRST half -- which modulo shapes are accepted
and which `shape:` slot each one lands in. See ptr_wrap_stacked.py and
ptr_wrap_side_by_side.py for the split that follows.

COVERS (6)   source root /workspace/triton_shared/, worktree lines

  rem-rank1-shape                   lib/AnalysisStructured/PtrAnalysis.cpp:855
      the remsi operand is rank 1 and has no prior modulo ->
      state.shape.back() = rhsState.scalar. Load 1 takes the modulo BEFORE the
      expand_dims, which is the ordinary matmul-row idiom.
  rem-rank2-dim1-singleton          PtrAnalysis.cpp:872
      rank 2 with result shape[1] == 1 -> state.shape[0] = the modulus.
      Load 2: `rm[:, None] % M` is a 16x1 tensor.
  rem-rank2-dim0-singleton          PtrAnalysis.cpp:870
      rank 2 with result shape[0] == 1 -> state.shape[1] = the modulus.
      Load 3: `rn[None, :] % M` is a 1x32 tensor. Same source expression, other
      axis, and it lands in the other slot -- the reason both ids exist.
  addstate-modulo-rhs-zero-offset   PtrAnalysis.cpp:477
      addState's modulo loop: lhs has modulo in dim i and rhs's offset for that
      dim is a constant zero -> keep lhs's modulo. This is what lets the column
      half be added to a row-modulo state at all.
  addstate-modulo-lhs-nomod         PtrAnalysis.cpp:475
      the other arm of that loop: dim 1 has no modulo, so shape[1] = 0 is
      copied through. (It also runs for every non-modulo add in every other
      kernel; it is documented here because this is where the `shape` list it
      writes is actually interesting.)
  addstate-splat-base-offset        PtrAnalysis.cpp, PtrState::settleBaseOffset
      adding a SPLAT SCALAR to a state that already has modulo in dim 0. The
      scalar belongs to no axis, so it goes to PtrState::baseOffset and is
      settled onto an axis that does NOT wrap; the modulus stays on the axis
      the program wrapped. Load 4 is `pw + BN`.

WHAT PROVES IT FIRED (01-structured.mlir, the four tts.make_tptr feeding the
tts.load ops -- `shape:` is the wrap list)

  load 1  ... strides: [%c32, 1], offsets: [%11, 0], shape: [256, 0]
  load 2  ... strides: [%c32_6, 1], offsets: [0, 0],  shape: [256, 0]
  load 3  ... strides: [%c32_7, 1], offsets: [0, 0],  shape: [0, %c8_8]
  load 4  ... strides: [%c32_6, 1], offsets: [0, %c32_9], shape: [256, 0]

  256 is the modulus 8 scaled by the `* BN` that follows it, so loads 1 and 2
  agree even though one took the rank-1 arm and the other the rank-2 arm.
  Load 3 puts 8 in the OTHER slot: the wrapped axis is the column, and the
  column term has no stride multiplier. Load 4 is the payoff -- it is load 2
  plus `+ 32` and nothing else, and its wrap STAYS in slot 0 while the `+ 32`
  lands on the axis that does not wrap. Offsets are absolute element offsets
  that the lowerings sum, so naming it on either axis is the same address --
  and only this placement leaves the modulus where the program put it.

  IT USED TO BE THE OTHER WAY. `offsets: [%c32_9, 0], shape: [0, 256]` -- the
  wrap swapped onto the column axis -- and triton-shared warned that it might
  miscompile, so this file's reference() returned zeros. Measured against the
  real math: 4.78054 max abs err before, 0 after.

  Every one of the four is `order: []` with at least one non-zero `shape`
  entry, i.e. tts::MakeTensorPtrOp::isSplitPtr(). There is no arith.remsi left
  in the module -- the modulus exists only as that shape operand.
  There is no warning for load 4 any more. The one that used to fire said
  "verify the operand that contains a scalar is meant to increment pointers in
  dim1 ... it WILL LEAD TO WRONG COMPILATION RESULTS", and it fired because the
  scalar had been parked on dim 0 with nowhere else to go.
      $TORCHSIM_COMPILE_TRITON_SHARED_OPT --triton-to-structured \
          --debug-only=triton-ptr-analysis \
          work/ptr_modulo_wrap_shapes_kernel/01-ttir.mlir -o /dev/null
      ptr_modulo_wrap_shapes.py:102:21: warning: PtrAnalysis: allowing adding
      pointer state with modulo in dim 0 to another pointer state with offset
      in dim 0. ... it WILL LEAD TO WRONG COMPILATION RESULTS.
  Line 102 is load 4. The swapped shape list above is visible without any flag.

COMPILE-TIME PROBE. The wrap-around semantics of load 4 are exactly the ones
the warning says may be wrong, so reference() does not try to predict them.

The four accepted modulo shapes and which `shape:` slot each writes. Load 4 pins that a splat scalar added to a dim-0-modulo state does NOT move the wrap: 4.78054 max abs err on a build without triton_shared 3500149, 0 with it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 8           # the modulus
BM = 16
BN = 32
N = 512


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_modulo_wrap_shapes_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                      BM: tl.constexpr, BN: tl.constexpr):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)

        # 1 modulo at rank 1, then expand_dims -> shape.back()
        rows = (pid * BM + rm) % M
        a = tl.load(x_ptr + rows[:, None] * BN + rn[None, :])

        # 2 modulo of a 16x1 tensor -> shape[0]
        pw = x_ptr + (rm[:, None] % M) * BN + rn[None, :]
        b = tl.load(pw)

        # 3 modulo of a 1x32 tensor -> shape[1]
        c = tl.load(x_ptr + rm[:, None] * BN + (rn[None, :] % M))

        # 4 splat scalar added to a dim-0-modulo state -> shape entries SWAPPED
        d = tl.load(pw + BN)

        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], a + b + c + d)

    return ptr_modulo_wrap_shapes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    i = torch.arange(BM).unsqueeze(1)
    j = torch.arange(BN).unsqueeze(0)
    a = x[(i % M) * BN + j]          # load 1: modulo at rank 1
    b = a                            # load 2: same address, other shape slot
    c = x[i * BN + (j % M)]          # load 3: modulo on the other axis
    d = x[(i % M) * BN + j + BN]     # load 4: + BN, OUTSIDE the modulo
    return {"o": (a + b + c + d).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_modulo_wrap_shapes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"M": M, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
