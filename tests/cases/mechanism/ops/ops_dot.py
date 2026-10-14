"""MatmulConverter -- all three accumulator arms of tt.dot in one module.

Split out of the old tests/cases/ops.py.  The three arms differ ONLY in
what the tt.dot's C operand is (splat-nonzero float / splat-nonzero int /
default zero), the setup is identical for all three, and telling them apart
means diffing three linalg.matmul sites in the same file -- which is exactly the
case rule 5b says to keep together.

COVERS -- 2 branch ids, plus one that belongs to another file (see below).
Line numbers are the triton-shared WORKING TREE, source root
/workspace/triton_shared/.

  dot-matmul-addc-f   lib/Conversion/TritonArithToLinalg/
                      TritonArithToLinalg.cpp:76
      MatmulConverter with skipC == false and a float result -> one
      linalg.matmul whose outs IS the dot's C (outs + A*B).

  dot-matmul-addc-i   TritonArithToLinalg.cpp:76
      the same converter with an integer result -> the same matmul into C.

  NOT CLAIMED HERE: dot-matmul-skipc.  The third tl.dot below does witness it
  (see the IR quote), but the branch census assigned that id to
  tests/cases/smoke/gemm.py, whose `acc += tl.dot(a, b)` can only ever produce it.  It is
  kept in this file anyway because without it the two addc arms have nothing to
  be different FROM: skipC is the shape the reader needs to see to understand
  what "addc" means.

WHY skipC IS NOT WHAT gemm.py's SOURCE LOOKS LIKE
  This is the mechanical reason the two addc arms cannot live in gemm.py.
  triton's own CombineDotAddFPattern (Combine.cpp:235-274, run by
  passes.ttir.add_combine) rewrites AddFOp(acc, DotOp(a, b, 0)) into
  DotOp(a, b, acc) BEFORE the linalg lowering sees anything.  So gemm's
  `acc += tl.dot(a, b)` arrives as a tt.dot whose C is the live scf.for
  iter-arg, isZeroTensor is false, and it takes dot-matmul-addc-f.  Writing the
  accumulator explicitly, as `d1` does here, is the only way to be sure which
  arm you get -- and conversely `dz` must be stored ALONE, because folding its
  result into an add would hand it straight back to that same pattern.

WHAT PROVES IT FIRED   work/ops_dot_kernel/triton-shared-detail/

  01-structured.mlir shows the three tt.dot with the three C operands.  Note
  %cst is dense<0.000000e+00>, which is exactly what isZeroTensor tests:

      %cst   = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
      %cst_1 = arith.constant dense<1> : tensor<32x32xi32>
      %cst_4 = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
      %25 = tt.dot %13, %24, %cst_4 : tensor<32x64xf32> * tensor<64x32xf32>
                                      -> tensor<32x32xf32>
      %30 = tt.dot %27, %29, %cst_1 : tensor<32x64xi8> * tensor<64x32xi8>
                                      -> tensor<32x32xi32>
      %41 = tt.dot %13, %24, %cst   : tensor<32x64xf32> * tensor<64x32xf32>
                                      -> tensor<32x32xf32>

  03-arith-linalg.mlir: the two addc arms are ONE linalg.matmul each, accumulating
  into their C -- no zero fill, no trailing add:

      %19 = linalg.matmul ins(%16, %18 : tensor<32x64xf32>, tensor<64x32xf32>)
                          outs(%14 : tensor<32x32xf32>)       <- C = splat 1.0
      %26 = linalg.matmul ins(%22, %25 : tensor<32x64xi8>, tensor<64x32xi8>)
                          outs(%12 : tensor<32x32xi32>)       <- C = splat 1

  and the third dot, the skipC arm, is the only one that starts from a zero fill:

      %30 = linalg.fill ins(%cst : f32) outs(%29 : tensor<32x32xf32>)
      %31 = linalg.matmul ins(%16, %18 ...) outs(%30 ...) -> tensor<32x32xf32>

  So the file contains 3 linalg.matmul and 1 linalg.fill feeding one, which is the
  whole claim in one sentence.  The two float dots share their operands (%16, %18)
  and are NOT CSE'd: their outs differ.

WHY THE SHAPES ARE 32x64 AND 64x32
  tl.dot needs every dimension >= 16, and the int8 arm needs K >= 32.  Making M
  (32) and K (64) DIFFERENT is deliberate: with M == K the two loads
  `x[i*K + j]` and `x[i*M + j]` would be the same tts.make_tptr and CSE into one
  operand, leaving a dot of a tensor with itself and one fewer thing to read.

ARITHMETIC
  o[0:1024]    = (x2 @ bm + 1) + (int8(8*x2) @ int8(8*bm) + 1)
  o[1024:2048] = x2 @ bm
  rtol/atol are 1e-3, the same as tests/cases/smoke/gemm.py: linalg.matmul accumulates in
  a different order from torch and f32 sums of 64 terms disagree in the last
  couple of bits.

SCOPE
  IR-census kernel.  Both claimed ids -- and dot-matmul-skipc, which gemm.py
  owns -- were confirmed by gdb line hits (the branch census), and the IR
  was read from tools/lower_steps.py.  Like every file under tests/cases/
  it does not reach run.py's numeric stages, so the rtol/atol below are the
  values this kernel WOULD need rather than ones a run has exercised.

POSITIVE KERNEL: every pass of the linalg lowering succeeds.

MatmulConverter's two non-zero-accumulator arms (float addf, int addi), with the default-accumulator skipC dot alongside for contrast (that id is claimed by tests/cases/smoke/gemm.py). POSITIVE.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 32
K = 64
XN = M * K       # 2048
ON = 2 * M * M   # 2048


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_dot_kernel(x_ptr, o_ptr, M: tl.constexpr, K: tl.constexpr):
        rows = tl.arange(0, M)
        cols = tl.arange(0, K)
        x2 = tl.load(x_ptr + rows[:, None] * K + cols[None, :])    # M x K
        bm = tl.load(x_ptr + cols[:, None] * M + rows[None, :])    # K x M

        # A. non-zero FLOAT accumulator -> linalg.matmul + arith.addf
        d1 = tl.dot(x2, bm, tl.full((M, M), 1.0, tl.float32))

        # B. non-zero INT accumulator -> linalg.matmul + arith.addi
        ai = (x2 * 8.0).to(tl.int8)
        bi = (bm * 8.0).to(tl.int8)
        di = tl.dot(ai, bi, tl.full((M, M), 1, tl.int32), out_dtype=tl.int32)

        tl.store(o_ptr + rows[:, None] * M + rows[None, :],
                 d1 + di.to(tl.float32))

        # C. DEFAULT accumulator: C is dense<0.0>, isZeroTensor is true, so
        #    MatmulConverter takes the skipC arm -- empty + fill + matmul with
        #    NO trailing accumulate generic.  It must be its own tt.dot and it
        #    must be stored ALONE: fold its result into an add and triton's
        #    CombineDotAddFPattern turns it back into an addc dot.
        dz = tl.dot(x2, bm)
        tl.store(o_ptr + M * M + rows[:, None] * M + rows[None, :], dz)

    return ops_dot_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(XN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    x2 = x.reshape(M, K)
    bm = x.reshape(K, M)
    d1 = x2 @ bm + 1.0
    di = ((x2 * 8.0).to(torch.int8).to(torch.int32)
          @ (bm * 8.0).to(torch.int8).to(torch.int32)) + 1
    o = torch.zeros(ON, dtype=torch.float32)
    o[0:M * M] = (d1 + di.to(torch.float32)).reshape(-1)
    o[M * M:2 * M * M] = (x2 @ bm).reshape(-1)
    return {"o": o}


SPEC = KernelSpec(
    name="ops_dot_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "K": "constexpr"},
    constexprs={"M": M, "K": K},
    args=[Arg("x", IN, "float32", (XN,)), Arg("o", OUT, "float32", (ON,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-3,
    atol=1e-3,
    expect="blocked",
    expect_why="the i8 arm: the matrix unit's words are f32, two f16 or two bf16 (vcix-accelerator examples/tpuv6, formats 0, 1 and 3), so there is no integer MAC to lower to",
)
