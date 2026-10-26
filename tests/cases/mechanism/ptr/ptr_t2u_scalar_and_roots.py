"""SCALAR pointers, and the two base roots that are not a kernel argument.

Everything else in this directory loads through a tensor of addresses. This file
is the other half of the unstructured pipeline: an address that is a single
number. It takes a different converter (affine.load / affine.store rather than
linalg.generic), and it is where TritonToUnstructured's three seeding rules
become visible -- kernel argument, scalar tt.int_to_ptr, tensor tt.int_to_ptr.

PART NEGATIVE, deliberately. The last load's base is a TENSOR of tt.int_to_ptr
results, which the pass refuses to seed (multiple bases are unsupported). That
tt.load is therefore left in the module verbatim, all the way to
05-unstruct-memref.mlir. That survival IS the observable for
t2u-seed-inttoptr-tensor-skip; do not try to make it lower.

WHY THESE LIVE TOGETHER: the scalar load/store pair and the two int_to_ptr
roots share the same mechanism -- TritonToUnstructured seeding a base and
accumulating a SCALAR i32 offset onto it. The scalar loads give the offset a
consumer (a scalar tts.gather), the ptr_to_int makes the pass rebuild a
tt.addptr from it, and the scalar int_to_ptr makes it land in a tts.make_tptr's
`offsets` instead. Three uses of one accumulator; separating them would leave
three kernels differing by one line.

COVERS (9)
  lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp
    t2u-offsettype-scalar-ptr    :199  getPtrOffsetType on a non-tensor
        !tt.ptr -> a plain i32 offset, so the gather prints `(<f32>, i32) -> f32`
    t2u-getbitwidth-scalar       :212  getBitWidth on a plain IntegerType (the
        scalar `si` offset), not on a tensor element type
    t2u-ptrtoint-scalar-materialize :302  a tt.addptr is REBUILT from
        (root, accumulated offset) and spliced in front of the tt.ptr_to_int --
        the only place this pass creates a tt.addptr
    t2u-seed-inttoptr-scalar     :273  a scalar tt.int_to_ptr becomes a second
        base root with its own `arith.constant 0 : i32`
    t2u-seed-inttoptr-tensor-skip :270  a TENSOR tt.int_to_ptr is skipped
    t2u-maketptr-offset-indexcast :530  the structured load off the int_to_ptr
        root: its tts.make_tptr offsets are `index`, so the accumulated i32 base
        offset is arith.index_cast'ed before being added in
  lib/Dialect/TritonStructured/IR/TritonStructuredOps.cpp
    gsv-scalar-passthrough       :74   `other` is already a scalar (the load is
        scalar), so getScalarValue returns it with no new ops
  A SCALAR tts.gather HAS NO LOWERING HERE, and that is why this kernel is
  refused. UnstructuredToMemref used to turn one into an affine.load and read
  NEITHER its mask nor its `other` -- a semantics-changing branch, and the
  reason a masked scalar load was not the same program after lowering. That
  pass is deleted (rule 21); b08 refuses instead, by name:
      Fatal: make_tptr base does not trace to a function arg
  A refusal that names itself beats a silent wrong answer (philosophy 12), so
  the two ids that used to live here are not a gap to fill.

HOW TO RUN IT
  `--to triton-compile` SUCCEEDS: stage 3 lowers tt.ptr_to_int / tt.int_to_ptr
  through tptr.from_memref / tptr.ptradd / tptr.inttoptr and writes all its
  files. Stage 4 is what refuses, by name:
      Fatal: make_tptr base does not trace to a function arg:
        OpResult(%19 = "tptr.inttoptr"(%18) : (i64) -> !ptr.ptr<...>)
  For the per-step IR the quotes below come from:
      python tools/lower_steps.py <this file> --debug

WHAT PROVES IT
  02-unstructured.mlir, 26 lines, verbatim -- note the SCALAR operand types:
      %1 = tts.gather %arg0[%arg7] : (<f32>, i32) -> f32
      %4 = tts.gather %arg0[%3] mask = %2 default = %arg8 : (<f32>, i32) -> f32
      tts.scatter %1 into %arg3[%arg7] : f32 into (<f32>, i32)
      tts.scatter %4 into %arg3[%3] mask = %2 : f32 into (<f32>, i32)
    `(<f32>, i32)` rather than `(<f32>, tensor<...xi32>)` is
    t2u-offsettype-scalar-ptr; `default = %arg8` is gsv-scalar-passthrough --
    the runtime `fill` ARGUMENT itself, with no splat and no rebuild in
    between. And the ptr_to_int:
      %5 = arith.addi %arg6, %arg7 : i32
      %6 = tt.addptr %arg0, %5 : !tt.ptr<f32>, i32
      %7 = tt.ptr_to_int %6 : !tt.ptr<f32> -> i64
    %6 did not exist in 01-structured.mlir; the pass built it from the root and
    the accumulated offset.
  The scalar int_to_ptr root and its index_cast, three lines apart:
      %0  = arith.index_cast %arg7 : i32 to index
      %9  = tt.int_to_ptr %8 : i64 -> !tt.ptr<f32>
      %11 = tts.make_tptr %9 to sizes: [128], strides: [1], offsets: [%0],
            shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
    A tts.make_tptr whose base is NOT a kernel argument, with the scalar
    offset spliced into its index-typed `offsets` -- that splice is :530.
  05-unstruct-memref.mlir -- the scalar load and store, and the dropped mask:
      %reinterpret_cast = memref.reinterpret_cast %m to offset: [%idx],
                          sizes: [1], strides: [1] : memref<*xf32>
                          to memref<1xf32, strided<[1], offset: ?>>
      %v = affine.load %reinterpret_cast[0] : memref<1xf32, ...>
      affine.store %v, %reinterpret_cast_N[0] : memref<1xf32, ...>
    There is no scf.if and no select anywhere near them, even though two of the
    four scalar ops were written with `mask=`.
  And the refusal, the last two lines before the store:
      %16 = tt.int_to_ptr %14 : tensor<128xi64> -> tensor<128x!tt.ptr<f32>>
      %17 = tt.load %16 : tensor<128x!tt.ptr<f32>>
    A raw tt.load, in a module where every other load has been converted.
    Compare %9 four lines up: the SCALAR tt.int_to_ptr did get seeded and now
    feeds a tts.make_tptr. Same op, same pass, opposite outcome, decided
    entirely by whether the result type is a RankedTensorType.

torch equivalent: compile-time probe (`reference` returns zeros). The addresses
read through `av`/`ad` are raw device addresses that only a real run could
satisfy, and one load is never lowered at all.

Scalar pointers (affine.load/store, mask silently dropped) plus the int_to_ptr and ptr_to_int roots. The tensor int_to_ptr load is left unlowered on purpose.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

NX = 4096
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_t2u_scalar_and_roots_kernel(x_ptr, ad_ptr, av_ptr, o_ptr, u_ptr,
                                        n, sh, si, fill, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)

        # -- scalar load/store: affine.load / affine.store, mask+other DROPPED
        sca = tl.load(x_ptr + si)
        scb = tl.load(x_ptr + si + 1, mask=si < n, other=fill)
        tl.store(o_ptr + si, sca)
        tl.store(o_ptr + si + 1, scb, mask=si < n)

        # -- scalar tt.ptr_to_int: the pass rebuilds a tt.addptr to feed it
        tl.store(u_ptr, (x_ptr + sh + si).to(tl.uint64))

        # -- scalar tt.int_to_ptr: a SECOND base root, seeded like an argument
        addr = tl.load(ad_ptr)
        ip = addr.to(tl.pointer_type(tl.float32))
        tl.store(o_ptr + 256 + r, tl.load(ip + si + r))

        # -- tensor tt.int_to_ptr: NOT seeded; this tt.load is never lowered
        addrs = tl.load(av_ptr + r)
        tl.store(o_ptr + 512 + r,
                 tl.load(addrs.to(tl.pointer_type(tl.float32))))

    return ptr_t2u_scalar_and_roots_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "ad": torch.zeros(1, dtype=torch.int64),
            "av": torch.zeros(BLOCK, dtype=torch.int64)}


def reference(inputs):
    import torch
    return {"o": torch.zeros(NX, dtype=torch.float32),
            "u": torch.zeros(1, dtype=torch.int64)}


SPEC = KernelSpec(
    name="ptr_t2u_scalar_and_roots_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "ad_ptr": "*i64", "av_ptr": "*i64",
               "o_ptr": "*fp32", "u_ptr": "*i64", "n": "i32", "sh": "i32",
               "si": "i32", "fill": "fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (NX,)), Arg("ad", IN, "int64", (1,)),
          Arg("av", IN, "int64", (BLOCK,)),
          Arg("o", OUT, "float32", (NX,)), Arg("u", OUT, "int64", (1,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t"), ("sh", "int32_t"),
                           ("si", "int32_t"), ("fill", "float")],
           "scalar_values": {"n": 64, "sh": 0, "si": 0, "fill": 0.0}},
    expect="refuse",
    expect_stop=Stop("transform", Reason.POINTER_BASE_UNTRACED),
    expect_why="part negative by design: the last load's base is a TENSOR of tt.int_to_ptr",
)
