#ifndef TORCHSIM_OPS
#define TORCHSIM_OPS

include "TorchSim/TorchSimDialect.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TorchSim_TransferOp : TorchSim_Op<"transfer",
    [DestinationStyleOpInterface, AttrSizedOperandSegments]> {
  let summary = "One DMA between DRAM and a scratchpad tile.";
  let description = [{
    The scratchpad side is the operand every axis reader relates; the DRAM side
    carries no layout, only `dram_stride`. A tensor tile and one result is the
    form above bufferization, a memref tile and no result the form below.
    Past the six fixed slots: the gather `index` (read at `offset_stride`), a
    [low, high] `clamp` per `masked_axes` entry, a runtime `dyn_fill`, one
    `dyn_strides` value per `dyn_stride_axes` entry, a runtime offset stride.
  }];
  let arguments = (ins AnyMemRef:$dram, Index:$dram_idx,
                       AnyType:$sram, Index:$sram_idx,
                       Index:$sem,
                       Optional<AnyShaped>:$index,
                       Variadic<Index>:$clamp,
                       Optional<I64>:$dyn_fill,
                       Variadic<I64>:$dyn_strides,
                       Optional<I64>:$dyn_offset_stride,
                       StrAttr:$dma_kind,
                       I64Attr:$dram_arg,
                       I64ArrayAttr:$dram_stride,
                       OptionalAttr<I64ArrayAttr>:$masked_axes,
                       OptionalAttr<I64Attr>:$masked_fill,
                       OptionalAttr<I64ArrayAttr>:$dyn_stride_axes,
                       OptionalAttr<I64Attr>:$offset_stride,
                       OptionalAttr<I64Attr>:$indirect_dim,
                       OptionalAttr<I64ArrayAttr>:$subtile_size,
                       OptionalAttr<TypedArrayAttrBase<DenseI64ArrayAttr,
                           "one cut entry per tile axis">>:$cut,
                       UnitAttr:$dram_bound,
                       OptionalAttr<BoolAttr>:$async);
  let results = (outs Optional<AnyType>:$tile);
  let hasVerifier = 1;
  let extraClassDeclaration = [{

    ::mlir::MutableOperandRange getDpsInitsMutable();
  }];
}

def TorchSim_DotOp : TorchSim_Op<"dot", [DestinationStyleOpInterface]> {
  let summary = "One product on the systolic array: init + a x b, batch axes leading.";
  let description = [{
    TritonGPU's tt.dot made this machine's own op where its layouts are decided: A, B and the
    init each carry the layout of the array's operands, which no GPU dot encoding states. The
    init is the destination; on buffers the product is added into it. `indexing_maps`, as
    linalg.matmul's, names which axis of each operand is m, n and k once a layout has
    permuted its axes; absent, the matmul definition ([..,M,K] x [..,K,N] -> [..,M,N]).
  }];
  let arguments = (ins AnyShaped:$a, AnyShaped:$b, AnyShaped:$init,
                       OptionalAttr<AffineMapArrayAttr>:$indexing_maps);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];
}

def TorchSim_WaitOp : TorchSim_Op<"wait"> {
  let summary = "Block until every transfer signalled on `sem` since its last wait has landed.";
  let description = [{
    A transfer signals `sem` with the elements it moves; one wait covers the whole batch since
    the previous wait on `sem`, in any completion order, and `amount` is that batch's sum.
  }];
  let arguments = (ins Index:$sem, Index:$amount);
  let hasVerifier = 1;
}

def TorchSim_CrossLaneMoveOp : TorchSim_Op<"crosslane_move"> {
  let summary = "One tile carried into another's lanes by the cross-lane unit.";
  let description = [{
    `torchsim.crosslane` names the XLU op and `torchsim.rounds` the plan it was
    priced at. Buffers only: lower-copies makes it from a `torchsim.copy` once
    every buffer is placed, so it never meets a tensor.
  }];
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
}

def TorchSim_CopyOp : TorchSim_Op<"copy", [DestinationStyleOpInterface]> {
  let summary = "A tile carried into another's lanes; which unit carries it is chosen below.";
  let description = [{
    What moves, never how, in LOGICAL axes (`*_perm`: the logical axis of each stored
    axis). A window of `src` (`static_*`), or of `dst` with `into_window`; `added_dims`
    broadcasts; neither: the same elements in row-major logical order.
  }];
  let arguments = (ins AnyType:$src, AnyType:$dst,
                       DenseI64ArrayAttr:$src_perm, DenseI64ArrayAttr:$dst_perm,
                       OptionalAttr<DenseI64ArrayAttr>:$static_offsets,
                       OptionalAttr<DenseI64ArrayAttr>:$static_sizes,
                       OptionalAttr<DenseI64ArrayAttr>:$static_strides,
                       UnitAttr:$into_window,
                       OptionalAttr<DenseI64ArrayAttr>:$added_dims);
  let results = (outs Optional<AnyType>:$copied);
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getDstMutable());
    }
  }];
}

def TorchSim_VcixOp : TorchSim_Op<"vcix", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<ConditionallySpeculatable>]> {
  let summary = "A VCIX instruction a kernel wrote by its encoding, on whole tiles in its layout.";
  let description = [{
    `form` is the VCIX op form (iv, v.i, v.iv), `opcode` and `imm` its fields, passed through:
    nothing here knows what the instruction does. `layouts` (one per operand) and `out_layout`
    are the kernel's statement of the TritonGPU encoding each tile takes; the anchor pins the
    types to them and nothing after reads them. One with no result or no operand reads and writes
    the state of the unit it names. One with both is assumed only to be a function of its
    operands -- nothing about what it computes. On tensors a result is the tile it yields; on
    buffers `dst` is the tile it writes.
  }];
  let arguments = (ins Variadic<AnyType>:$args, Optional<AnyMemRef>:$dst, I64Attr:$opcode,
                       I64Attr:$imm, StrAttr:$form, ArrayAttr:$layouts,
                       OptionalAttr<AnyAttr>:$out_layout);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Whether it reads or writes its unit's state: it has no result, or no operand.
    bool isStateful();
  }];
}

def TorchSim_BankViewOp : TorchSim_Op<"bank_view", [Pure]> {
  let summary = "A buffer seen with the strides a lane's bank really has.";
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$view);
}

def TorchSim_VLaneIdxOp : TorchSim_Op<"vlane_idx", [Pure]> {
  let summary = "The lane number, from the hardware and never from a loop.";
  let results = (outs AnyVectorOfAnyRank:$idx);
}

def TorchSim_StridedLoadOp : TorchSim_Op<"strided_load"> {
  let summary = "One vector of `count` elements `stride` apart from `indices`: a vlse.";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$base, Variadic<Index>:$indices,
                       I64Attr:$stride, I64Attr:$count);
  let results = (outs AnyVectorOfAnyRank:$result);
}

def TorchSim_StridedStoreOp : TorchSim_Op<"strided_store"> {
  let summary = "One vector into `count` elements `stride` apart from `indices`: a vsse.";
  let arguments = (ins AnyVectorOfAnyRank:$value, Arg<AnyMemRef, "", [MemWrite]>:$base,
                       Variadic<Index>:$indices, I64Attr:$stride, I64Attr:$count);
}

def TorchSim_SplatLoadOp : TorchSim_Op<"splat_load"> {
  let summary = "The one element at `indices` of each lane's bank, in every element of a vector: "
                "what a value with no extent along the vector reads as. Never a scalar: the lanes "
                "share the scalar registers.";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$base, Variadic<Index>:$indices);
  let results = (outs AnyVectorOfAnyRank:$result);
}

def TorchSim_ReductionOp : TorchSim_Op<"reduction", [Pure, SameVariadicOperandSize]> {
  let summary = "Vectors folded to one element each, then combined with `accs`: a vector.reduction "
                "with an accumulator whose combiner is the body, its ins the fold and outs the accs.";
  let arguments = (ins Variadic<AnyVectorOfAnyRank>:$vectors, Variadic<AnyVectorOfAnyRank>:$accs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$combiner);
}

def TorchSim_YieldOp : TorchSim_Op<"yield", [Pure, Terminator]> {
  let summary = "What a reduction's combiner hands back.";
  let arguments = (ins Variadic<AnyType>:$values);
}

def TorchSim_FusionFenceOp : TorchSim_Op<"fusion_fence",
                                         [Pure, SameOperandsAndResultType]> {
  let summary = "A value handed straight back, so nothing fuses across it.";
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$fenced);
}

def TorchSim_DmaOp : TorchSim_Op<"dma"> {
  let summary = "The trace's DMA: one togsim_dma call, signalling `sem` when it is an async load.";
  let arguments = (ins Index:$dram_idx, Index:$sem);
}

def TorchSim_ComputeOp : TorchSim_Op<"compute"> {
  let summary = "The trace's compute: one togsim_compute call.";
}

def TorchSim_MemoryBarrierOp : TorchSim_Op<"memory_barrier"> {
  let summary = "The trace's barrier: waits the batch signalled on `sem`, whose sum is `expected`.";
  let arguments = (ins Index:$sem, I64Attr:$expected);
  let hasVerifier = 1;
}

#endif
