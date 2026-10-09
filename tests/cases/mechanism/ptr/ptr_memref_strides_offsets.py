"""The four ways a structured tts.make_tptr's layout can be written, and what
each one becomes in a torchsim.transfer.

Split out of the old tests/cases/ptr_structured.py section D. Nothing here is
about PtrAnalysis -- every address in this file is trivially structured. The
subject is the (offsets, strides, sizes) triple: when an entry is a literal and
when it is an SSA operand, and what a size-1 axis with a zero stride does.

This file used to be about StructuredToMemref, which turned that triple into a
`strided<[...], offset: ...>` layout on a memref.reinterpret_cast. That pass is
deleted (rule 21) and p06 reads the triple itself, so the same five loads now
pin p06's _dram_strides and its dynamic-operand slots instead.

COVERS (4)   pytorchsim_triton_compiler/lowering/passes/p01_convert_accesses_to_transfer.py

  stride-static               _dram_strides, the plain arm
      a constant stride is carried through as an int. Loads 3, 4 and 6:
      static_strides = [32, 1] -> dram_stride = [32, 1].
  stride-dynamic              _dram_strides, the Value arm
      the stride is an SSA Value, so static_strides carries kDynamic and the
      transfer names the axis instead. Load 2, `* sy` with sy an i32 kernel
      argument:
          static_strides = [-9223372036854775808, 1]
          -> dram_stride = [0, 1], dyn_stride_axes = [0], one extra operand
  stride-size1-zero           _dram_strides, the size-1 arm
      size == 1 AND the stride is a constant 0 -> the ACCUMULATED product of
      the inner dims, not 0. Load 1 is `x_ptr + r[None, :]`:
          sizes = [1, 32], static_strides = [0, 1] -> dram_stride = [32, 1]
      No zero-stride transfer is ever emitted, and the unit axis survives
      --linalg-fold-unit-extent-dims (the scratchpad is memref<1x32xf32, 1>).
  offset-static-vs-dynamic    the dram_offset operand
      a constant offset is %c0; a computed one is an SSA index. Loads 4 and 5
      are the same address with and without a `pid *` term, and only load 5
      passes a value there.

WHAT PROVES IT (08-transformed.mlir, verbatim -- seven transfers, one per access)

  load 1, size-1 dim -> stride 32 synthesised, NOT 0:
      "torchsim.transfer"(%reinterpret_cast_2, %c0, %6, %c0, %7, %c0)
        {dma_kind = "MVIN", dram_arg = 0, dram_stride = [32, 1]}
        : (memref<256xf32>, index, memref<1x32xf32, 1>, ...) -> ()

  load 2, dynamic stride -> the axis is named, the attribute holds a placeholder:
      {dram_stride = [0, 1], dyn_stride_axes = [0]}   with %9 as the operand

transfer layouts: static/dynamic strides, the size-1 zero-stride fixup, static vs dynamic offset. Runtime i32 scalar sy=32. Numerically runnable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BS = 32
NX = 256
NY = BS * BS
SY = 32          # the runtime stride value


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_memref_strides_offsets_kernel(x_ptr, y_ptr, o1_ptr, o2_ptr, sy,
                                          BS: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BS)

        s = tl.load(x_ptr + r[None, :])                     # 1 size-1 dim
        d = tl.load(y_ptr + r[:, None] * sy + r[None, :])   # 2 runtime stride
        e = tl.load(y_ptr + r[:, None] * BS + r[None, :])   # 3 constant stride
        f = tl.load(x_ptr + r)                              # 4 static offset
        g = tl.load(x_ptr + pid * BS + r)                   # 5 dynamic offset

        tl.store(o2_ptr + r[:, None] * BS + r[None, :], d + e + s)
        tl.store(o1_ptr + r, f + g)

    return ptr_memref_strides_offsets_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "y": torch.randn(NY, generator=g, dtype=torch.float32)}


def reference(inputs):
    x, y = inputs["x"], inputs["y"]
    tile = y.reshape(BS, BS)                     # sy == BS, so 2 and 3 agree
    s = x[0:BS].reshape(1, BS)
    return {"o1": x[0:BS] + x[0:BS],             # pid == 0
            "o2": (tile + tile + s).reshape(NY)}


SPEC = KernelSpec(
    name="ptr_memref_strides_offsets_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o1_ptr": "*fp32",
               "o2_ptr": "*fp32", "sy": "i32", "BS": "constexpr"},
    constexprs={"BS": BS},
    args=[Arg("x", IN, "float32", (NX,)), Arg("y", IN, "float32", (NY,)),
          Arg("o1", OUT, "float32", (BS,)), Arg("o2", OUT, "float32", (NY,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"sy": SY}},
)
