"""tts.gather + tts.scatter, unmasked -- the first scatter that ever reached Spike.

The backend read tts.scatter's operands in PRINTED order. The custom assembly
prints `tts.scatter %value into %ptr[%offset]`, but the ODS argument order is

    (ins TT_Ptr:$ptr, TT_IntLike:$offset, TT_Type:$value, Optional:$mask)

so operands[1] is the OFFSET and operands[2] is the VALUE -- the two were
swapped. Here the offsets are i32 and the value f32, so the swap surfaced as
`bufferization.to_buffer (tensor<128x16xi32>) -> memref<128x16xf32>` in the
verifier. It would have gone through SILENTLY for a scatter whose value and
offsets share an element type (an i32 tensor scattered by i32 indices), which is
why this file checks the numbers rather than just that it compiles.

SHAPE IS NOT ARBITRARY
  dim0 must be the lane count. At 8x16 this same kernel dies in Spike with a
  load segfault -- see the dense sweep in the small-tile note: any rank>=2 tile
  with dim0 < VECTORLANE_SIZE is mis-addressed by the pad/slice bridge, gather
  or not.

torch equivalent: o[idx] = x[idx] * 2.

tts.scatter operand order (ptr, offset, value) -- was swapped, and would have been silent for same-typed value/offset tensors.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 8192
BM, BN = 128, 16        # dim0 == VECTORLANE_SIZE, see above


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_scatter_roundtrip_kernel(x_ptr, i_ptr, o_ptr,
                                     BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        # a BMxBN index block: both dims > 1, so there is no single gather dim
        # and PtrAnalysis has to emit the fully unstructured tts.gather/scatter
        uidx = tl.load(i_ptr + rm[:, None] * BN + rn[None, :])
        v = tl.load(x_ptr + uidx)
        tl.store(o_ptr + uidx, v * 2.0)

    return ptr_scatter_roundtrip_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "i": torch.randperm(N, generator=g)[:BM * BN].to(torch.int32)}


def reference(inputs):
    import torch
    o = torch.zeros(N, dtype=torch.float32)
    idx = inputs["i"].long()
    o[idx] = inputs["x"][idx] * 2.0
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_scatter_roundtrip_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (N,)), Arg("i", IN, "int32", (BM * BN,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
