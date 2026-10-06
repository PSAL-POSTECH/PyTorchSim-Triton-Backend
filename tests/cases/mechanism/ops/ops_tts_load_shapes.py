"""Every operand slot of tts.make_tptr and tts.load, in one shape each.

Split out of the old tests/cases/ptr_structured.py section F. Nothing here
is a new compiler DECISION -- the branches this file covers are ODS facts about
the two ops (TritonStructuredDialect.td), not C++ `if`s. That is exactly why
they need their own file: they are verified by READING the printed op, and in a
kernel with sixty loads there is no way to point at the one op that shows a
given slot. Here there are five loads and each one differs from the others in
exactly one slot.

NOT GDB-MEASURABLE. the branch census cannot set a breakpoint in a
TableGen file, so it counts every id below as ".td, no code" and leaves it out
of the target denominator. The evidence is the quoted IR, and the way to
re-check it is:
    python run.py tests/cases/ops_tts_load_shapes.py --to triton-compile
    python tools/lower_steps.py tests/cases/ops_tts_load_shapes.py
    grep 'tts.make_tptr\\|tts.load' \\
        work/ops_tts_load_shapes_kernel/triton-shared-detail/01-structured.mlir

COVERS (13)   include/triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.td

  tts-mktptr-structured          :115  isStructuredPtr(): order empty AND every
                                       static_shape entry is 0. Loads 1 and 2.
  tts-mktptr-sizes-always-static :58   `sizes` is a bare DenseI64ArrayAttr with
                                       NO variadic counterpart -- a dynamic
                                       block size cannot be expressed at all
                                       (PtrState::createTTSMakeTensorPtrOp
                                       asserts on it).
  tts-mktptr-strides-static      :62   constant stride -> literal in the
                                       DynamicIndexList, no operand. Load 1.
  tts-mktptr-strides-dynamic     :59   runtime stride -> an index operand and
                                       kDynamic in static_strides. Load 2.
  tts-mktptr-offsets-static      :63   base offset folds to a literal. Load 1.
  tts-mktptr-offsets-dynamic     :60   `pid *` makes it an index operand.
                                       Load 2.
  tts-mktptr-shape-static        :64   `shape: [0]`, the "no wrap in this dim"
                                       literal. All five loads.
                                       (shape-dynamic needs a modulo -- it is
                                       claimed by ptr_wrap_stacked.py.)
  tts-load-ptr-is-tensor-of-ptrs :313  TT_PtrLike:$ptr resolving to
                                       tensor<...x!tt.ptr<T>>. All five.
  tts-load-no-mask               :331  both mask_dims and static_mask_dims
                                       empty -> hasMask() false. Loads 1, 2.
  tts-load-mask-dims-static      :315  static_mask_dims holds real constants
                                       and mask_dims is empty. Load 3.
  tts-load-mask-dims-dynamic     :314  the mask_dims variadic is present and
                                       static_mask_dims holds kDynamic.
                                       Loads 4, 5.
  tts-load-other-present         :316  the optional $other -- note it is a
                                       SCALAR (f32 here), not a tensor;
                                       rewriteLoadOp reduces the tensor `other`
                                       to a scalar with getScalarValue and
                                       FAILS if it is not uniform. Loads 3, 4.
  tts-load-other-absent          :316  omitted -> the third field of
                                       operandSegmentSizes is 0. Loads 1, 2, 5.

WHAT PROVES IT FIRED (01-structured.mlir) -- one line per slot

  load 1  everything literal, nothing optional:
      %3 = tts.make_tptr %arg0 to sizes: [32], strides: [1], offsets: [0],
           shape: [0], order: [] : <f32> to tensor<32x!tt.ptr<f32>>
      %5 = "tts.load"(%3) <{operandSegmentSizes = array<i32: 1, 0, 0>,
           static_mask_dims = array<i64>}>
           : (tensor<32x!tt.ptr<f32>>) -> tensor<32xf32>
  load 2  both index lists switch to operands (sizes cannot -- there is no
  variadic to switch to):
      %13 = tts.make_tptr %arg0 to sizes: [32], strides: [%12], offsets: [%7],
           shape: [0], order: [] : <f32> to tensor<32x!tt.ptr<f32>>
  load 3  a constant clamp, no index operand, `other` present:
      %17 = "tts.load"(%3, %cst_2) <{operandSegmentSizes = array<i32: 1, 0, 1>,
           static_mask_dims = array<i64: 24>}>
           : (tensor<32x!tt.ptr<f32>>, f32) -> tensor<32xf32>
  load 4  a runtime clamp AND `other`:
      %23 = "tts.load"(%3, %22, %cst_3) <{operandSegmentSizes = array<i32: 1, 1, 1>,
           static_mask_dims = array<i64: -9223372036854775808>}>
           : (tensor<32x!tt.ptr<f32>>, index, f32) -> tensor<32xf32>
  load 5  the same, minus `other` -- the only difference is the last 1 becoming
  a 0 in operandSegmentSizes:
      %29 = "tts.load"(%3, %28) <{operandSegmentSizes = array<i32: 1, 1, 0>,
           static_mask_dims = array<i64: -9223372036854775808>}>
           : (tensor<32x!tt.ptr<f32>>, index) -> tensor<32xf32>
  Loads 1, 3, 4 and 5 all reuse the SAME %3 pointer: the four tts.load ops
  differ only in the slots this file is about.

torch equivalent: see reference(). Runtime scalars stride=1, nkeep=12,
nfull=32.

ODS shapes of tts.make_tptr / tts.load, one slot per load. Verified by reading 01-structured.mlir -- .td ids are outside what gdb can measure. Runtime i32 scalars stride=1, nkeep=12, nfull=32.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 256
BLOCK = 32
STRIDE = 1
NKEEP = 12
NFULL = 32
CONSTBOUND = 24


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_tts_load_shapes_kernel(x_ptr, o_ptr, stride, nkeep, nfull,
                                   BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, BLOCK)

        a = tl.load(x_ptr + r)                              # 1 all static
        b = tl.load(x_ptr + pid * BLOCK + r * stride)       # 2 all dynamic
        c = tl.load(x_ptr + r, mask=r < 24, other=0.0)      # 3 static dims
        d = tl.load(x_ptr + r, mask=r < nkeep, other=-1.0)  # 4 dynamic + other
        e = tl.load(x_ptr + r, mask=r < nfull)              # 5 dynamic, no other

        tl.store(o_ptr + r, a + b + c + d + e)

    return ops_tts_load_shapes_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    r = torch.arange(BLOCK)
    base = x[0:BLOCK]                          # pid == 0, stride == 1
    z = torch.zeros(BLOCK, dtype=torch.float32)
    v = base + base
    v = v + torch.where(r < CONSTBOUND, base, z)
    v = v + torch.where(r < NKEEP, base, torch.full_like(base, -1.0))
    v = v + base                               # nfull covers the whole block
    return {"o": v}


SPEC = KernelSpec(
    name="ops_tts_load_shapes_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "stride": "i32",
               "nkeep": "i32", "nfull": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_values": {"stride": STRIDE, "nkeep": NKEEP, "nfull": NFULL}},
)
