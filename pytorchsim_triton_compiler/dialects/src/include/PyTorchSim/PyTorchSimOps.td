#ifndef TORCHSIM_OPS
#define TORCHSIM_OPS

include "PyTorchSim/PyTorchSimDialect.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def PyTorchSim_TransferOp : PyTorchSim_Op<"transfer",
    [DestinationStyleOpInterface, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "One DMA between DRAM and a scratchpad tile.";
  let description = [{
    `direction` is "load" (DRAM into the scratchpad) or "store". The scratchpad side is
    the operand every axis reader relates; the DRAM side carries no layout, only `dram_stride`. A tensor tile and one result is the
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
                       StrAttr:$direction,
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

def PyTorchSim_DotOp : PyTorchSim_Op<"dot", [DestinationStyleOpInterface,
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

def PyTorchSim_GatherOp : PyTorchSim_Op<"gather", [DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Each element of a tile taken from another along one axis, at the place an index tile names.";
  let description = [{
    Triton's tt.gather: `init[i] = src[i with i[axis] = index[i]]`, src, index and init one shape
    and, once laid out, one layout. `axis` is a stored axis once the layouts are made into shapes.
    Along the lane axis the cross-lane unit carries it (the index rows the pattern each lane reads
    by); along an axis inside each lane's bank, each lane's registers do.
  }];
  let arguments = (ins AnyShaped:$src, AnyShaped:$index, AnyShaped:$init, I64Attr:$axis);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];
}

def PyTorchSim_WaitOp : PyTorchSim_Op<"wait", [PyTorchSim_Landing]> {
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
                           [MemRead<PyTorchSim_InFlight>, MemWrite<PyTorchSim_InFlight>]>:$landing);
  let hasVerifier = 1;
}

def PyTorchSim_DistributedReductionOp : PyTorchSim_Op<"distributed_reduction"> {
  let summary = "A reduction across the distributed axis: every element of a column combined into one row.";
  let description = [{
    `src` is spread over the distributed axis (axis 0 of its bank shape); `dst` is one row every
    lane holds (UNIFORM). Column by column, the elements of `src` along the distributed axis are
    combined by `by` (an arith op, as it spells itself) and the result combined with what `dst`
    held. Which unit does it is the target's (slot lower-distributed-reduction). Buffers only:
    vectorize makes it from a linalg.reduce over the distributed axis once every buffer is placed. A
    reduction that stays within each element's holder is pytorchsim.reduction.
  }];
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$src, Arg<AnyMemRef, "", [MemRead, MemWrite]>:$dst,
                       StrAttr:$by);
}

def PyTorchSim_ConvertLayoutOp : PyTorchSim_Op<"convert_layout", [DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "A tile carried into another layout; which unit carries it is the pack's choice.";
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
  let results = (outs Optional<AnyType>:$converted);
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getDstMutable());
    }
  }];
}

def PyTorchSim_LaneIdOp : PyTorchSim_Op<"lane_id", [Pure, PyTorchSim_LaneVariant]> {
  let summary = "The lane number, from the hardware and never from a loop.";
  let results = (outs AnyVectorOfAnyRank:$idx);
}

def PyTorchSim_StridedLoadOp : PyTorchSim_Op<"strided_load"> {
  let summary = "One vector of `count` elements `stride` apart from `indices`: a vlse.";
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$base, Variadic<Index>:$indices,
                       I64Attr:$stride, I64Attr:$count);
  let results = (outs AnyVectorOfAnyRank:$result);
}

def PyTorchSim_StridedStoreOp : PyTorchSim_Op<"strided_store"> {
  let summary = "One vector into `count` elements `stride` apart from `indices`: a vsse.";
  let arguments = (ins AnyVectorOfAnyRank:$value, Arg<AnyMemRef, "", [MemWrite]>:$base,
                       Variadic<Index>:$indices, I64Attr:$stride, I64Attr:$count);
}

def PyTorchSim_ReductionOp : PyTorchSim_Op<"reduction", [Pure, SameVariadicOperandSize]> {
  let summary = "Vectors folded to one element each, then combined with `accs`: a vector.reduction "
                "with an accumulator whose combiner is the body, its ins the fold and outs the accs.";
  let arguments = (ins Variadic<AnyVectorOfAnyRank>:$vectors, Variadic<AnyVectorOfAnyRank>:$accs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$combiner);
}

def PyTorchSim_YieldOp : PyTorchSim_Op<"yield", [Pure, Terminator]> {
  let summary = "What a reduction's combiner hands back.";
  let arguments = (ins Variadic<AnyType>:$values);
}

def PyTorchSim_FusionFenceOp : PyTorchSim_Op<"fusion_fence",
                                         [Pure, SameOperandsAndResultType]> {
  let summary = "A value handed straight back, so nothing fuses across it.";
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$fenced);
}

def PyTorchSim_RuntimeAttributeOp : PyTorchSim_Op<"runtime_attribute",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "A value the program reads from data and its TOG must know: recorded by the run, read by the TOG.";
  let description = [{
    `form` says which side of record `id` (of `rank` dims) this is. "S" records `value`, read
    outside every loop, and yields it. "A" records, before a loop nest, the elements
    `source[base + sum(t_d * strides_d)]` for every trip `t` below `extents`. "G" records the
    index tile `source` at trip `indices` of the `extents` leading trips, just before the
    indirect transfer it addresses. "R" yields `value`, element `indices` (trips) of an "A"
    record, where the loop reads it. `capacity` bounds each dim of an "A" or "G" record.
  }];
  let arguments = (ins Optional<AnySignlessIntegerOrIndex>:$value,
                       Optional<AnyMemRef>:$source,
                       Optional<Index>:$base,
                       Variadic<Index>:$indices,
                       Variadic<Index>:$extents,
                       I32Attr:$id, I32Attr:$rank, StrAttr:$form,
                       OptionalAttr<DenseI64ArrayAttr>:$strides,
                       OptionalAttr<DenseI64ArrayAttr>:$capacity);
  let results = (outs Optional<AnySignlessIntegerOrIndex>:$result);
  let hasVerifier = 1;
}

#endif
