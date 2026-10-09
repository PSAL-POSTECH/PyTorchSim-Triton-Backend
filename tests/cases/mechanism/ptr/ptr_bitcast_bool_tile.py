"""a MASKED load through a `*i1` pointer -- the unstructured load keeps its
`other=`, and the gather takes it once it is reduced to a scalar.

REDUCED FROM GPT-2 (tests/models/GPT2/test_gpt2.py --preset tiny), kernel 5
`triton_npu_fused__safe_softmax_add_scalar_tensor_slice_view_where_5`, the
causal-mask + softmax.  Kernels 0-4 compile; this is where the model stops.

THE CHAIN, and every link is needed

  1. the causal mask is a `torch.bool` buffer, so Inductor hands the kernel a
     `*i1` argument.
  2. triton's `tl.load` on an i1 pointee bitcasts the POINTER TENSOR first --
     `tt.bitcast : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>` -- because
     i1 occupies a byte in memory.
  3. PtrAnalysis cannot walk THROUGH a bitcast on a tensor of pointers, so the
     access never becomes a `tts.make_tptr`/`tts.load`.  It falls to the
     unstructured path: one scalar `tt.load` per element inside a
     `linalg.generic`.  (`--triton-to-unstructured` prints
     `error: unexpected op in ptr sequence` here and then EXITS 0, so stage 3
     reports ok -- do not read that message as the failure.)
  4. Inductor's load is masked, so the scalarised load carries THREE operands.
     Measured, 07-regularized.mlir:56

       %28 = "tt.load"(%arg16, %arg17, %arg18)
               <{operandSegmentSizes = array<i32: 1, 1, 1>}>
               : (!tt.ptr<i8>, i1, i8) -> i8

  5. p00_convert_unsupported_pointers used to refuse it outright -- "a tensor tt.load
     carrying `other=` is not wired; it would be silently dropped" -- because
     the gather's fill is a SCALAR that _empty_read_mask_carriers splats back, while an
     unstructured load's `other` is a TILE.

WHAT THIS PINS
  Those two are the SAME statement whenever the tile is uniform, and triton
  only ever spells a scalar fill as a tile because a tensor load has no scalar
  operand to put it in.  `uniform_scalar` (lib_mlir_util, the dual of `splat`)
  reduces the tile back to its one value and convert_unsupported_pointers hands it to the gather`s
  `other` slot, which existed and had never been wired.  Measured here:
  linalg.fill(%c0_i8) -> tts.gather(..., mask, %c0_i8), exact through spike.

  A tile that genuinely VARIES per element is still refused, by a message of
  its own -- the gather cannot hold it and splatting one of its values would be
  the silent wrong answer.  ptr_gather_other_varies.py pins that half.

WHAT BREAKS THIS KERNEL -- i.e. what it is guarding
  Drop `mask=`/`other=` from the `m_ptr` load and it passes for a DIFFERENT
  reason, pinning nothing: that was measured, the first draft of this file was
  unmasked and went green on the unfixed tree.  So the bitcast alone is not
  what this guards.  Equally, make `m_ptr` an `*i8` and the load structures
  normally into a `tts.load`, which has taken `other` as an operand all along.
  Both halves have to stay for this to be about the unstructured path.

A *i1 pointee makes triton bitcast the POINTER TENSOR, which PtrAnalysis cannot walk, so the load goes unstructured and takes its `other` tile with it. convert_unsupported_pointers reduces that uniform tile to the scalar the gather's fill slot wants. Reduced from GPT-2 kernel 5.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128
LIVE = 100          # < BLOCK, so the mask really masks and `other` is reachable


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_bitcast_bool_tile_kernel(m_ptr, x_ptr, o_ptr, LIVE: tl.constexpr,
                                     BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        ok = r < LIVE
        # *i1 -> triton bitcasts the POINTER TENSOR to !tt.ptr<i8>, which the
        # structured analysis cannot walk, so this load lands on the
        # unstructured (per-element) path -- carrying its `other` with it.
        m = tl.load(m_ptr + r, mask=ok, other=0)
        x = tl.load(x_ptr + r, mask=ok, other=0.0)
        tl.store(o_ptr + r, tl.where(m, x, 0.0))

    return ptr_bitcast_bool_tile_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"m": (torch.rand(N, generator=g) > 0.5),
            "x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    live = torch.arange(N) < LIVE
    return {"o": torch.where(inputs["m"] & live, inputs["x"],
                             torch.zeros_like(inputs["x"]))}


SPEC = KernelSpec(
    name="ptr_bitcast_bool_tile_kernel",
    kernel=kernel,
    signature={"m_ptr": "*i1", "x_ptr": "*fp32", "o_ptr": "*fp32",
               "LIVE": "constexpr", "BLOCK": "constexpr"},
    constexprs={"LIVE": LIVE, "BLOCK": BLOCK},
    args=[Arg("m", IN, "bool", (N,)),
          Arg("x", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
)
