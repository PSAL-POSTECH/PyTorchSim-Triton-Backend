"""A modulo meeting a gather: which of the two axes it lands on decides.

PtrAnalysis can encode `offs % M` as a wrap-around pointer (`shape: [M]`, the
"split pointer"; p06 states its runs and p16 builds one DMA piece per lap).
That encoding requires the dim to be AFFINE. When the `%` sits on the axis that
GATHERS it cannot stay a wrap and the whole side is parked as the index instead
(loads 1, 2, 4, 5); when it sits on any OTHER axis it keeps its own slot on the
gather pointer (load 3, since the slot was added).

Five loads, one theme. They differ in WHERE the modulo sits relative to the
loaded index, which is exactly what selects the arm.

COVERS (7)   lib/AnalysisStructured/PtrAnalysis.cpp
  rem-analysing-unstructured-rebuild   PtrAnalysis.cpp:808
      visitOperandRem in the SECOND (unstructured) pass returns immediately
      with rebuildAsUnsupportedOp(remOp.getResult()) -- it never looks at the
      operands.  [load 1]
  rem-lhs-unstructured-rebuild-gs      PtrAnalysis.cpp:833
      the FIRST pass: after visiting the lhs the state is not structured, so
      rebuildAsGatherScatter(remOp.getResult(), nonStructuredDim).  [load 1]
  rebuild-gs-success                   PtrAnalysis.cpp:265
      the body of that rebuild: offsets[dim] = the remsi result, strides[dim] =
      1, shape[dim] = 0 -- the modulo is CLEARED.  [loads 1, 2]
  rem-multi-modulo-rank1-gs            PtrAnalysis.cpp:847
      nested modulo `(r % M) % 7`: the "do not support multiple modulo" remark
      is still printed, but rank == 1 and gather/scatter is enabled, so the
      rewrite succeeds anyway as a gather.  [load 2]
  addstate-modulo-nonstructured-shape0 PtrAnalysis.cpp:469
      addState's modulo loop, per dim: a dim that is not structured on either
      side gets `shape` pushed as the constant 0 while the OTHER dim keeps its
      real modulo shape.  [load 3]
  add-lhs-unstruct-rhs-modulo-rebuild  PtrAnalysis.cpp:738
      visitOperandAdd: lhs not structured AND rhs hasModulo -> the RHS is
      rebuilt as a gather, discarding its modulo.  [load 4]
  add-lhs-modulo-rhs-unstruct-rebuild  PtrAnalysis.cpp:746
      the mirror: lhs hasModulo AND rhs not structured -> the LHS is rebuilt.
      Loads 4 and 5 differ ONLY in the order of the two addends, which is the
      whole reason they live in one file: the emitted gather_scatter_offset is
      the arith.addi's lhs in one and its rhs in the other.  [load 5]

WHAT PROVES IT (01-structured.mlir)
  Load 1  x[i % M]:
      %6 = arith.remsi %4, %cst : tensor<128xi32>
      %7 = tts.make_gather_scatter_tptr %arg0 to sizes: [128]
           gather_scatter_dim: 0 gather_scatter_offset: %6, strides: [1],
           offsets: [0] : tensor<128xi32>  <f32> to !tt.ptr<tensor<128xf32>>
    The remsi RESULT is the offset operand, and the op has no `shape:` field at
    all (tts.make_gather_scatter_tptr cannot express wrap-around). Compare a
    structured `x[r % M]`, which is a tts.make_tptr with `shape: [64]`.
  Load 2  x[(r % M) % 7] -- with --debug-only=triton-ptr-analysis:
      remark: PtrAnalysis: do not support multiple modulo within an expression
    and yet a gather_scatter_tptr on the nested remsi is emitted anyway. The
    remark is not a failure here; that is the point of :847.
  Load 3  x[i[:, None]*16 + (rn % M)[None, :]] -- one dim modulo, one dim
    gather, on DIFFERENT axes. `shape: [0, %M]`: the gather dim's entry is 0
    because a gathered axis wraps inside its index, the other dim's is the real
    modulus. THE WRAP HERE IS INERT -- rn is 16 wide and M is 64 -- so this arm
    pins the encoding and not the arithmetic; ptr_wrap_meets_gather is where it
    actually fires.
  Loads 4 and 5 both produce
      %N = arith.addi %a, %b : tensor<8xi32>
      ... gather_scatter_offset: %N
    but in load 4 %a is the modulo term and %b the loaded index, and in load 5
    they are swapped -- read the two arith.addi operand orders to tell which
    arm ran.

WHY IT MATTERS
  `% M` on an affine offset is nearly free (it becomes a pointer attribute).
  `% M` anywhere near a loaded index costs a full elementwise arith.remsi plus
  the per-element gather loop. Nothing in the Triton source distinguishes the
  two; these five lines are what the distinction looks like.

STATUS: green, both outputs err = 0.

  o   loads 1-2, rank 1.  The modulo survives even though
      rebuildAsGatherScatter clears the `shape` slot: the remsi RESULT becomes
      the gather offset, so it is computed at runtime. Semantically equivalent,
      not a loss -- which is why the reference below is the real computation and
      not the zeros this file used to carry.

  o2  loads 3-5, rank 2.  Was wrong by 5.06 for most of the work on this file,
      and the cause was neither of the two things it looked like. NOT the
      wraparound the `shape:` discussion above points at -- load 3 reproduces
      the error alone, and its wrap is inert at these sizes. NOT a rank
      restriction either; lifting those changed nothing, twice.

      It was the vector axis. Load 3's offset is built by a generic whose body
      uses `linalg.index 1` -- the position INSIDE a lane's vector, which the
      backend had no way to name until vector.step (-> vid.v) was wired up.
      `rn` alone would have folded into a DMA stride; `rn % M` cannot, so the
      column index has to exist as a real number, sixteen at a time.

torch equivalent: see `reference` -- the real computation. Several of
these addresses are deliberately degenerate and the file exists for the IR.

Every place a modulo meets a non-structured dim and loses its wrap-around encoding.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
NI = 1024
M = 64
BM = 8
BN = 16
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_modulo_kernel(x_ptr, i_ptr, o_ptr, o2_ptr,
                                 M: tl.constexpr, BM: tl.constexpr,
                                 BN: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        i = tl.load(i_ptr + r)                  # 128 loaded indices
        i8 = tl.load(i_ptr + rm)                # 8 loaded indices
        k8 = tl.load(i_ptr + 32 + rm)           # 8 more, so loads 4/5 differ

        # 1. modulo OF a loaded index -> remsi result becomes the gather offset
        a = tl.load(x_ptr + (i % M))
        # 2. nested modulo of an affine offset: remark, then gather anyway
        b = tl.load(x_ptr + ((r % M) % 7))
        # 3. modulo on the NON-gather dim of a row gather
        c = tl.load(x_ptr + i8[:, None] * BN + (rn % M)[None, :])
        # 4. add(unstructured, modulo): the RHS is rebuilt
        d = tl.load(x_ptr + (i8[:, None] + (rm % M)[:, None]) * BN
                    + rn[None, :])
        # 5. add(modulo, unstructured): the LHS is rebuilt -- same two terms,
        #    swapped
        e = tl.load(x_ptr + ((rm % M)[:, None] + k8[:, None]) * BN
                    + rn[None, :])

        tl.store(o_ptr + r, a + b)
        tl.store(o2_ptr + rm[:, None] * BN + rn[None, :], c + d + e)

    return ptr_gather_modulo_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, M, (NI,), generator=g, dtype=torch.int32)}


def reference(inputs):
    import torch
    x, i = inputs["x"], inputs["i"]
    r = torch.arange(BLOCK)
    rm, rn = torch.arange(BM), torch.arange(BN)
    i128, i8 = i[:BLOCK].long(), i[:BM].long()
    k8 = i[32:32 + BM].long()
    a = x[(i128 % M)]
    b = x[((r % M) % 7)]
    c = x[i8[:, None] * BN + (rn % M)[None, :]]
    d = x[(i8[:, None] + (rm % M)[:, None]) * BN + rn[None, :]]
    e = x[((rm % M)[:, None] + k8[:, None]) * BN + rn[None, :]]
    return {"o": a + b, "o2": (c + d + e).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_gather_modulo_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "o2_ptr": "*fp32", "M": "constexpr", "BM": "constexpr",
               "BN": "constexpr", "BLOCK": "constexpr"},
    constexprs={"M": M, "BM": BM, "BN": BN, "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (NI,)),
          Arg("o", OUT, "float32", (BLOCK,)),
          Arg("o2", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
