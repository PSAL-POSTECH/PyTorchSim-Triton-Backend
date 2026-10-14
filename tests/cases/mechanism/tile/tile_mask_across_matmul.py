"""An i1 buffer is written bit-packed and read one byte per element.

THE MACHINE STORES A MASK REGISTER PACKED. `affine.vector_store` of a
`vector<8xi1>` into a `memref<64xi1>` becomes

    vsm.v  v8, (a2)        one BIT per element -- 8 elements, 1 byte
    addi   a2, a2, 8       the pointer moves 8 BYTES to the next 8 elements

so seven bytes in every eight are never written, and a load of a different width
reads the wrong ones back:

    vlm.v  v0, (a2)        32 elements out of 4 bytes
    addi   a2, a2, 32

Everything else in a spad buffer is one element per byte or wider, and the DMA
copies the buffer densely, so an i1 buffer is consistent only with itself and
only at one vector width. This kernel makes a DMA read one.

IT FAILED BY 1920 of 8192 elements -- exactly the fifteen rows the mask should
have stopped -- until p04_bufferize._widen_i1_buffers gave the predicate a byte
per element. Run it on a tree without that and it comes back.

resnet18's sixteenth conv builds

    idx_n = (nhw // OUT_W) // OUT_H ;  keep = idx_n < BATCH

before its accumulation loop and stores under it afterwards. The scratchpad
cannot hold the 64-element predicate across the loop, so place_in_spad spills it and
reads it back BROADCAST to the [64, 128] tile:

    MVOUT memref<64xi1>    -> dram scratch,   dram_stride [1]
    MVIN  memref<64x128xi1> <- dram scratch,  dram_stride [1, 0]

MEASURED on that conv, the mask that comes back is true for nhw = 0..55 when it
must be true only for 0..48 -- the seven rows where idx_n is 1 and nh is exactly
OUT_H come back live, so they store over output row h = 0 and the real values
there are lost. idx_n itself is right: the gather offsets for those same rows
carry idx_n * stride_xn = 50176.

WHAT DISTINGUISHES THIS FROM tile_mask_from_nested_div.py, which passes: there
the whole predicate is one elementwise chain and never becomes a buffer, so no
DMA sees it. The matmul is what forces the spill, and i1 is the element type the
other two spilled operands (idx_y_h, idx_y_w, both i32) do not have.

A store mask spilled to DRAM scratch as i1 across a matmul and read back broadcast comes home wrong; the i32 operands beside it do not.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, BN, BK, OUT_W, OUT_H, BATCH = 64, 128, 16, 7, 7, 1
KDEPTH = 2


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_across_matmul_kernel(x_ptr, w_ptr, o_ptr, BM: tl.constexpr,
                                  BN: tl.constexpr, BK: tl.constexpr,
                                  KDEPTH: tl.constexpr, OUT_W: tl.constexpr,
                                  OUT_H: tl.constexpr, BATCH: tl.constexpr):
        nhw = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        rk = tl.arange(0, BK)
        # NOT AFFINE IN nhw, so PtrAnalysis cannot fold it into a descriptor box
        # and it stays a mask operand -- the same shape resnet's conv has.
        idx_n = (nhw // OUT_W) // OUT_H
        keep = (idx_n < BATCH)[:, None] & (rn < BN)[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(KDEPTH):
            a = tl.load(x_ptr + nhw[:, None] * (BK * KDEPTH)
                        + (rk + k * BK)[None, :])
            b = tl.load(w_ptr + (rk + k * BK)[:, None] * BN + rn[None, :])
            acc += tl.dot(a, b, allow_tf32=True)
        tl.store(o_ptr + nhw[:, None] * BN + rn[None, :], acc, mask=keep)

    return mask_across_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BM * BK * KDEPTH, generator=g, dtype=torch.float32),
            "w": torch.randn(BK * KDEPTH * BN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(BM, BK * KDEPTH)
    w = inputs["w"].reshape(BK * KDEPTH, BN)
    acc = x @ w
    keep = torch.tensor([((m // OUT_W) // OUT_H) < BATCH for m in range(BM)])
    return {"o": torch.where(keep[:, None], acc, torch.zeros_like(acc)).reshape(-1)}


SPEC = KernelSpec(
    name="mask_across_matmul_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "o_ptr": "*fp32",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr",
               "KDEPTH": "constexpr", "OUT_W": "constexpr",
               "OUT_H": "constexpr", "BATCH": "constexpr"},
    constexprs={"BM": BM, "BN": BN, "BK": BK, "KDEPTH": KDEPTH,
                "OUT_W": OUT_W, "OUT_H": OUT_H, "BATCH": BATCH},
    args=[Arg("x", IN, "float32", (BM * BK * KDEPTH,)),
          Arg("w", IN, "float32", (BK * KDEPTH * BN,)),
          Arg("o", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
