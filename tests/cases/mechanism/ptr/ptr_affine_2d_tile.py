"""A 2-D tile address -- how a rank-2 PtrState is assembled out of rank-1 parts.

Split out of the old tests/cases/ptr_structured.py section A. `rm[:, None]
* BN + rn[None, :]` is the single most common address in all of Triton, and it
is also the only place four different PtrState combinators run one after the
other: expand_dims inserts a size-1 dim, broadcast widens it, mulState scales a
dim by a scalar, addState merges two rank-2 states dim by dim.

COVERS (5)   source root /workspace/triton_shared/, worktree lines

  expanddims-insert          lib/AnalysisStructured/PtrAnalysis.cpp:1024
      visitOperandExpandDims inserts offset 0 / size 1 / stride 0 / shape 0 at
      `axis`. Both `[:, None]` and `[None, :]` go through it.
  broadcast-size-expand      PtrAnalysis.cpp:1067
      visitOperandBroadcast, srcShape[i] < dstShape[i] -> sizes[i] = dstShape[i]
      while the stride for that dim stays 0. Broadcasting is encoded as a
      ZERO-STRIDE dimension, which is why load 2 below has `strides: [1, 0]`.
  broadcast-size-equal       PtrAnalysis.cpp:1065
      the other arm of the same loop: srcShape[i] == dstShape[i] -> continue,
      that dim of the state is untouched. Every tt.broadcast here hits both
      arms, one per dim.
  mulstate-structured-dim-mul  PtrAnalysis.cpp:573
      mulState per-dim: offsets, strides AND shape are all multiplied by the
      scalar. `rm[:, None] * BN` is what turns stride 1 into stride BN.
  addstate-structured-dim-add  PtrAnalysis.cpp:305
      addState per-dim with both sides structured: offsets summed, strides
      summed. The row state (stride [BN, 0]) plus the column state
      (stride [0, 1]) is what produces the final `strides: [BN, 1]`.

WHAT PROVES IT FIRED (01-structured.mlir)

      %2 = tts.make_tptr %arg0 to sizes: [16, 32], strides: [32, 1],
           offsets: [0, 0], shape: [0, 0], order: []
           : <f32> to tensor<16x32x!tt.ptr<f32>>
      %4 = tts.make_tptr %arg0 to sizes: [16, 32], strides: [1, 0],
           offsets: [0, 0], shape: [0, 0], order: []
           : <f32> to tensor<16x32x!tt.ptr<f32>>

  Load 1 (%2): `strides: [32, 1]` is the ADDED result -- 32 from
  mulstate-structured-dim-mul on the row half, 1 from the column half, summed
  by addstate-structured-dim-add. Load 2 (%4) is the broadcast on its own:
  `strides: [1, 0]`, and that literal 0 in dim 1 IS broadcast-size-expand's
  observable -- 32 columns that all read the same element. sizes are [16, 32]
  in both, so the expand_dims/broadcast pair reached the right rank.
  There is no arith.muli or tt.broadcast left feeding a pointer: all of it
  folded into the two attribute lists.

torch equivalent: o = x.reshape(16, 32) + x[0:16, None].expand(16, 32)

Rank-2 PtrState assembly: expand_dims, broadcast, mulState, addState. Numerically runnable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM = 16
BN = 32
N = BM * BN


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_affine_2d_tile_kernel(x_ptr, o_ptr, BM: tl.constexpr,
                                  BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)

        # 1 the ordinary tile: expand_dims + broadcast + mul + add
        t = tl.load(x_ptr + rm[:, None] * BN + rn[None, :])
        # 2 broadcast alone, so the zero-stride dim is visible by itself
        e = tl.load(x_ptr + tl.broadcast_to(rm[:, None], (BM, BN)))
        tl.store(o_ptr + rm[:, None] * BN + rn[None, :], t+e)

    return ptr_affine_2d_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"]
    tile = x.reshape(BM, BN)
    bcast = x[0:BM].reshape(BM, 1).expand(BM, BN)
    return {"o": (tile + bcast).reshape(N)}


SPEC = KernelSpec(
    name="ptr_affine_2d_tile_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BM": "constexpr",
               "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
