#ifndef TORCHSIM_OPS
#define TORCHSIM_OPS

include "TorchSim/TorchSimDialect.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def TorchSim_Semaphore : Resource<"::torchsim::Semaphore">;
def TorchSim_InFlight : Resource<"::torchsim::InFlight">;

// An op whose result differs between lanes with no operand or bank to say so: the lane number.
def TorchSim_LaneVariant : NativeOpTrait<"LaneVariant"> { let cppNamespace = "::torchsim::OpTrait"; }

defvar TorchSim_Landing = MemoryEffects<[MemRead, MemWrite, MemRead<TorchSim_Semaphore>,
                                         MemWrite<TorchSim_Semaphore>]>;

def TorchSim_TransferOp : TorchSim_Op<"transfer",
    [DestinationStyleOpInterface, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "One DMA between DRAM and a scratchpad tile.";
  let description = [{
    The scratchpad side is the operand every axis reader relates; the DRAM side
    carries no layout, only `dram_stride`. A tensor tile and one result is the
    form above bufferization, a memref tile and no result the form below.
    Past the six fixed slots: the gather `index` (read at `offset_stride`), a
    [low, high] `clamp` per `masked_axes` entry, a runtime `dyn_fill`, one
    `dyn_strides` value per `dyn_stride_axes` entry, a runtime offset stride.
    A `cut` states an axis's DRAM offset as digits, outer to inner, each a
    (radix, stride, modulus, phase) record: the outermost radix 0 is the rest
    of the axis, a modulus 0 an affine digit, and the axis's `dram_stride` is
    its inner digit's. A `piece_box` moves only that box of the tile,
    `dram_idx` addressing its first element; a `cut` is what
    fit-transfers-to-descriptor turns into such pieces, so the two never meet.
    A piece clamped on an axis its box bounds is a fill piece: the clamp is
    static and misses the box, so every element of the box takes the fill.
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
                           "one digit list per tile axis, empty where it is uncut">>:$cut,
                       OptionalAttr<TypedArrayAttrBase<DenseI64ArrayAttr,
                           "one [lo, hi] per tile axis, empty where the whole axis moves">>:$piece_box,
                       UnitAttr:$dram_bound,
                       OptionalAttr<BoolAttr>:$async);
  let results = (outs Optional<AnyType>:$tile);
  let hasVerifier = 1;
  let extraClassDeclaration = [{

    ::mlir::MutableOperandRange getDpsInitsMutable();
  }];
}

def TorchSim_DotOp : TorchSim_Op<"dot", [DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

def TorchSim_WaitOp : TorchSim_Op<"wait", [TorchSim_Landing]> {
  let summary = "Block until every transfer signalled on `sem` since its last wait has landed.";
  let description = [{
    A transfer signals `sem` with the elements it moves; one wait covers every transfer signalled
    on `sem` since its previous wait, in any completion order, and `amount` is their sum.
    `landing` names the scratchpad buffers those transfers move into or out of: what the wait reads
    and writes of them is the transfers in flight there (InFlight), not one element of their
    data; it is where their lives must still reach.
  }];
  let arguments = (ins Index:$sem, Index:$amount,
                       Arg<Variadic<AnyMemRef>, "",
                           [MemRead<TorchSim_InFlight>, MemWrite<TorchSim_InFlight>]>:$landing);
  let hasVerifier = 1;
}

def TorchSim_CrossLaneMoveOp : TorchSim_Op<"crosslane_move"> {
  let summary = "One tile carried into another's lanes by the cross-lane unit.";
  let description = [{
    `torchsim.crosslane` names the XLU op and `torchsim.rounds` the plan it was
    priced at. Buffers only: lower-copies makes it from a `torchsim.copy` once
    every buffer is placed, so it never meets a tensor.
  }];
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$src, Arg<AnyMemRef, "", [MemWrite]>:$dst);
}

def TorchSim_CopyOp : TorchSim_Op<"copy", [DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

def TorchSim_BankViewOp : TorchSim_Op<"bank_view",
    [Pure, DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "A buffer seen with the strides a lane's bank really has.";
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$view);
}

def TorchSim_PackOp : TorchSim_Op<"pack",
    [Pure, DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "A lane's bank read as words of 32 bits, as many of its elements to a word as fit.";
  let arguments = (ins AnyMemRef:$bank);
  let results = (outs AnyMemRef:$words);
}

def TorchSim_VLaneIdxOp : TorchSim_Op<"vlane_idx", [Pure, TorchSim_LaneVariant]> {
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

def TorchSim_DmaOp : TorchSim_Op<"dma", [TorchSim_Landing]> {
  let summary = "A DMA of the tile-operation-graph (TOG): one togsim::load or togsim::store call, signaling `sem` when it is an async load.";
  let description = [{
    `extents` size the dims its `dyn_dims` attribute names at run time: what a mask leaves live.
  }];
  let arguments = (ins Index:$dram_idx, Index:$sem, Variadic<Index>:$extents);
}

def TorchSim_ComputeOp : TorchSim_Op<"compute", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "The TOG's compute: one togsim::compute call.";
}

def TorchSim_MemoryBarrierOp : TorchSim_Op<"memory_barrier", [TorchSim_Landing]> {
  let summary = "The TOG's barrier: one togsim::wait call, waiting every load that signaled `sem` since its last wait.";
  let arguments = (ins Index:$sem);
}

#endif
