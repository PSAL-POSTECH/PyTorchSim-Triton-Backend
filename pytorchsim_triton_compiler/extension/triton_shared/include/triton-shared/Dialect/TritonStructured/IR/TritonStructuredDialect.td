//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_STRUCTURED_DIALECT
#define TRITON_STRUCTURED_DIALECT

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Triton_Structured_Dialect : Dialect {
  let name = "tts";

  let cppNamespace = "::mlir::tts";

  let summary = "Structured Triton operations";

  let description = [{
    Triton Structured Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect"
  ];

  let usePropertiesForAttributes = 1;
}

//
// Op Base
//
class TTS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Structured_Dialect, mnemonic, traits> {
}

def TTS_MakeTensorPtrOp
  : TTS_Op<"make_tptr", [AttrSizedOperandSegments, Pure]> {
  let summary = "create a pointer that points to a tensor in memory";

  // base:    Base pointer used to contruct the tensor of pointers or pointer to tensor.
  // sizes:   Size of the data being loaded or stored.
  // strides: The strides of the parent tensor, which means how much to increase the pointer
  //          by when moving by 1 element in a specific axis.
  // order:   The order of the block, which means how the block is laid out in memory.
  //          It contains the same info as order in tt.make_tensor_ptr.
  // shape:   If order is present, this field signifies the shape of the parent tensor in
  //          memory; if order is not present, it signifies the boundary by which addresses
  //          wraps around (constant zero indicates no wrap-around in the corresponding dimension).
  // offsets: Offset of the block along each dimension from base.
  // result:  If order is present, this op produces a pointer to a tensor; otherwise,
  //          it produces a tensor of pointers.

  let arguments = (ins TT_Ptr:$base,
                       DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$shape,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_shape,
                       // AXIS d'S INDEX IS `p // static_step[d]`, and 0 means
                       // no step -- the same spelling `shape` uses for no wrap.
                       // AN ATTRIBUTE AND NOT AN OPERAND, because a step whose
                       // period is not known here cannot have its runs counted
                       // and is refused rather than carried. That keeps the
                       // operand segments where every reader already expects
                       // them, and keeps a stepless op printing as it always
                       // did.
                       OptionalAttr<DenseI64ArrayAttr>:$static_step,
                       DenseI32ArrayAttr:$order,
                       // An address term added AFTER the wrap. Every axis that
                       // wraps computes `(offset + j*stride) mod shape`, so a
                       // term folded onto one is taken modulo too and cancels
                       // when it is a multiple of the wrap. It rides here when
                       // no axis is free to hold it -- see
                       // PtrState::createTTSMakeTensorPtrOp.
                       Optional<Index>:$base_offset,
                       // A modulus on the SUM of every axis, applied after each
                       // axis's own wrap. `shape` is a per-axis fact; this one
                       // belongs to no axis, which is why it rides beside
                       // base_offset instead of in that array.
                       Optional<Index>:$flat_shape);

  let results = (outs TT_PtrLike:$result);

  let assemblyFormat = [{
    $base `to` `sizes` `` `:` $sizes
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    `` `,` `offsets` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `shape` `` `:`
    custom<DynamicIndexList>($shape, $static_shape)
    `` `,` `order` `` `:` $order
    `` custom<TailSlots>($base_offset, $flat_shape)
    attr-dict `:` type($base) `to` type($result)
  }];


  let builders = [
    // Build with mixed static and dynamic entries.
    OpBuilder<(ins
      "Value":$base,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$shape,
      "ArrayRef<int32_t>":$order,
      CArg<"Value", "Value()">:$base_offset,
      CArg<"Value", "Value()">:$flat_shape,
      CArg<"Attribute", "Attribute()">:$encoding)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedSizes() {
      Builder b(getContext());
      SmallVector<Value> dynSizes; // sizes are always static
      return ::mlir::getMixedValues(getSizes(), dynSizes, b);
    }
    SmallVector<OpFoldResult> getMixedStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticStrides(), getStrides(), b);
    }
    SmallVector<OpFoldResult> getMixedOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(), b);
    }
    SmallVector<OpFoldResult> getMixedShape() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticShape(), getShape(), b);
    }
    bool isBlockPtr() {
      return !getOrder().empty();
    }
    bool isStructuredPtr() {
      return !isBlockPtr() && !getFlatShape() &&
             llvm::all_of(getStaticShape(), [](auto shape) { return shape == 0; });
    }
    SmallVector<int64_t> getSteps() {
      if (auto st = getStaticStep())
        return SmallVector<int64_t>(*st);
      return SmallVector<int64_t>(getSizes().size(), 0);
    }
    bool hasStep() {
      auto st = getStaticStep();
      return st && llvm::any_of(*st, [](int64_t v) { return v != 0; });
    }
    bool isSplitPtr() {
      return !isBlockPtr() &&
             !isStructuredPtr();
    }
  }];

  // A WRAP THAT CANNOT FIRE IS NOT A WRAP, and this op is where one would be
  // written down. Forbidden rather than folded: the analysis that builds this
  // consumes it instead (visitOperandRemAsWrap), and both ask one predicate.
  let hasVerifier = 1;
  // TODO
  //let hasCanonicalizer = 1;
}

def TT_IndexTensorLike : AnyTypeOf<[I32Tensor, I64Tensor]>;

def TTS_MakeGatherScatterTensorPtrOp
  : TTS_Op<"make_gather_scatter_tptr", [AttrSizedOperandSegments, Pure]> {
  // NOTE: Only support cases where the offset for each dimension is defined in a different operation.
  //       Not support case where the offset is a tensor load from other ptr which for multiple dimension.
  //
  //       offset_m = tl.arange(0, M)
  //       offset_n = tl.arange(0, N)
  //       offset_k = tl.arange(0, K)
  //       ld_offsets = tl.load(a_ptr + offset_m[:,None]+offsets_n[None,:])
  //       not_support = tl.load(b_ptr + ld_offsets)
  //       not_support2 = tl.load(b_ptr + ld_offsets * (offset_m[:,None]+offsets_n[None,:]))
  //       not_support3 = tl.load(b_ptr + (ld_offsets * (offset_m[:,None]+offsets_n[None,:]))[:, :, None] + offset_k[None,None,:])
  //       
  //       # Support cases where one dimension is structured while the other is not.
  //       # For example, `offset_m[:, None] // K` is not structured, whereas `offset_n[None, :]` is structured in next line.
  //       supported = tl.load(b_ptr + offset_m[:, None] // K + offset_n[None, :])
  //       # `ld_offsets_1d[:, None]` is not structured, whereas `offset_n[None, :]`.
  //       ld_offsets_1d = tl.load(a_ptr + offsets_m)
  //       supported2 = tl.load(b_ptr + ld_offsets_1d[:, None] + offsets_n[None, :])

  let summary = "create an pointer that points to a tensor in memory for gather/scatter";
  let description = [{
    The `tts.make_gather_scatter_tptr` operation is similar to `tts.make_tptr`.
    The key difference is that `make_gather_scatter_tptr` accesses the tensor non-continuously.
    Currently, only one dimension is allowed to be non-continuous.
    This dimension is saved in `gather_scatter_dim`, and the offset for that dimension is saved in `gather_scatter_offset`.
    Each contiguous load will load from this offset.
    Cases with more than one non-continuous dimension are not supported.
  }];

  // base:    Base pointer used to contruct the tensor of pointers or pointer to tensor.
  // gather_scatter_offset: The offset for gather/scatter.
  // gather_scatter_dim: The dimension for gather_scatter_offset.
  // sizes:   Size of the data being loaded or stored.
  // strides: The strides of the parent tensor, which means how much to increase the pointer
  //          by when moving by 1 element in a specific axis.
  // offsets: Offset of the block along each dimension from base.
  // gather_scatter_mask: Optional bool mask for mask which failed MaskAnalysis.
  // result:  A tensor of pointers.

  let arguments = (ins TT_Ptr:$base,
                       TT_IndexTensorLike:$gather_scatter_offset,
                       I32Attr:$gather_scatter_dim,
                       DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       // A wrap on an AFFINE axis. The gathered axis never has
                       // one -- its `%` is arithmetic inside the index tensor --
                       // so shape[gather_scatter_dim] is always 0.
                       Variadic<Index>:$shape,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_shape,
                       // ROW GRAIN: one bit per gathered row. tts.load's
                       // elem_mask is ELEMENT grain, and the two are kept
                       // apart on purpose -- see that slot for the price.
                       Optional<TT_BoolLike>:$gather_scatter_mask,
                       // An address term that belongs to no axis. A fully
                       // unstructured state has no axis to fold it onto: the
                       // gather dim's own offset is overwritten by the
                       // lowering, so it has to ride here as a value.
                       Optional<Index>:$base_offset);

  let results = (outs TT_PtrLike:$result);

  let assemblyFormat = [{
    $base `to` `sizes` `` `:` $sizes
    `gather_scatter_dim` `` `:` $gather_scatter_dim
    `gather_scatter_offset` `` `:` $gather_scatter_offset
    (`gather_scatter_mask` `` `:` $gather_scatter_mask^)?
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    `` `,` `offsets` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `shape` `` `:`
    custom<DynamicIndexList>($shape, $static_shape)
    (`` `,` `base_offset` `` `:` $base_offset^)?
    attr-dict `:`  type($gather_scatter_offset) type($gather_scatter_mask) type($base) `to` type($result)
  }];


  let builders = [
    // Build with mixed static and dynamic entries.
    OpBuilder<(ins
      "Value":$base,
      "Value":$gather_scatter_offset,
      "int":$gather_scatter_dim,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$shape,
      CArg<"Value", "Value()">:$base_offset)>,

    OpBuilder<(ins
      "Value":$base,
      "Value":$gather_scatter_offset,
      "Value":$gather_scatter_mask,
      "int":$gather_scatter_dim,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$shape,
      CArg<"Value", "Value()">:$base_offset)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedSizes() {
      Builder b(getContext());
      SmallVector<Value> dynSizes; // sizes are always static
      return ::mlir::getMixedValues(getSizes(), dynSizes, b);
    }
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticStrides(), getStrides(), b);
    }
    SmallVector<OpFoldResult> getMixedOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(), b);
    }
    SmallVector<OpFoldResult> getMixedShape() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticShape(), getShape(), b);
    }
    /// True when some AFFINE axis wraps. Static 0 is the only spelling of
    /// "does not wrap", so a dynamic extent counts even if it is 0 at runtime.
    bool hasWrap() {
      return llvm::any_of(getStaticShape(), [](int64_t s) { return s != 0; });
    }
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 0;
}

def TTS_GetStructuredStateOp : TTS_Op<"get_structured_state", [AttrSizedResultSegments, Pure]> {
  let summary = "Placeholder for the structured pointer states computed during PtrAnalysis.";
  let description = "Used to pass the offsets and strides to scf.for op to simplify IR rewrites.";

  // TT_IndexTensorLike, not I32Tensor: the prepass that BUILDS this op wraps
  // every int-or-index tensor in a loop's iter-args (see
  // TritonToStructuredPass's converter and getOffsetAndStrideSegmentSizes,
  // which both say "we only care about tensor of index / int"), so an i64
  // offset tensor reaches an op that used to refuse it. The two halves
  // disagreed, and the op was the narrow one -- MakeGatherScatterTensorPtrOp
  // above already takes i32 or i64 for the same kind of value.
  //
  // Reached by Qwen2-VL's masked_scatter, whose running index is int64: a
  // tl.associative_scan carried across the reduction loop, then used as a
  // gather offset. Before this the pass failed the whole kernel with
  // "operand #0 must be ptr or ranked tensor of ptr values or tensor of
  // 32-bit signless integer values, but got 'tensor<128x1xi64>'".
  let arguments = (ins AnyTypeOf<[TT_PtrLike, TT_IndexTensorLike]>:$input);
  // THE BASE OFFSET IS A THIRD COMPONENT, and it has to be one: a loop carries
  // a pointer by carrying its components, and a term that lives OUTSIDE the
  // wrap has no axis offset to ride in. Its count comes from the TYPE like the
  // other two, so every structured pointer gets one whether it needs it or
  // not -- exactly as a rank-2 pointer gets two strides it may never vary.
  let results = (outs AnyTypeOf<[TT_PtrLike, TT_IndexTensorLike]>:$structured, Variadic<Index>:$offsets, Variadic<Index>:$strides, Variadic<Index>:$base_offsets);

  let builders = [
    OpBuilder<(ins "Value":$input)>,
  ];

  let extraClassDeclaration = [{
    static std::optional<std::pair<SmallVector<Type>, SmallVector<Type>>>
      getOffsetAndStrideTypes(MLIRContext *context, Type ptrLikeType);

    static std::optional<std::pair<int32_t, int32_t>>
      getOffsetAndStrideSegmentSizes(Type ptrLikeType);

    /// 1 when this type's state can carry a base offset, 0 otherwise. Derived
    /// from the type for the same reason the other two are.
    static int32_t getBaseOffsetSegmentSize(Type ptrLikeType);
  }];

  let hasFolder = 0;
  let hasVerifier = 1;
}

def TTS_GatherOp : TTS_Op<"gather", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments,
  OptionalTypesMatchWith<"mask type matches ptr type", "offset", "mask", "triton::getI1SameShape($_self)">,
  OptionalTypesMatchWith<"other matches ptr type", "ptr", "other", "triton::getPointeeType($_self)">
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  let arguments = (
    ins
    TT_Ptr:$ptr,
    TT_IntLike:$offset,
    Optional<TT_BoolLike>:$mask,
    Optional<TT_Type>:$other
  );

  let results = (outs TT_Type:$result);

  let assemblyFormat = [{
    $ptr `[` $offset `]` (`mask` `=` $mask^)? (`default` `=` $other^)?
    attr-dict `:` `(` type($ptr) `,` type($offset) `)` `->` type($result)
  }];
}

def TTS_ScatterOp : TTS_Op<"scatter", [
  MemoryEffects<[MemWrite]>,
  OptionalTypesMatchWith<"mask type matches offset type", "offset", "mask",
                 "triton::getI1SameShape($_self)">
]> {
  let summary = "optionally store data from in memory to fill a portion of the tensor";

  let arguments = (
    ins
    TT_Ptr:$ptr,
    TT_IntLike:$offset,
    TT_Type:$value,
    Optional<TT_BoolLike>:$mask
  );

  let assemblyFormat = [{
    $value `into` $ptr `[` $offset `]` (`mask` `=` $mask^)?
    attr-dict `:` type($value) `into` ` ` `(` type($ptr) `,` type($offset) `)`
  }];
}

def TTS_LoadOp : TTS_Op<"load", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments,
  OptionalTypesMatchWith<"elem_mask is an i1 of the result's shape", "result",
                         "elem_mask", "triton::getI1SameShape($_self)">
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  // elem_mask: a mask MaskAnalysis could not fold into `mask_dims`, one bit per
  // element. It says which elements take part, not where they live -- the
  // pointer stays whatever it was, wrap and order included. How it is realised
  // (a select on the loaded tile, a clamp, a park) is the consumer's choice.
  //
  // ITS TWIN IS make_gather_scatter_tptr's gather_scatter_mask, one bit per
  // gathered ROW, and folding that one into this one is a trade, not a tidy-up:
  // it costs every consumer that reads a row-masked index one entry per row
  // instead of one per element. Measured 2026-09-02: 44 of 45 masked gather
  // pointers in the 430-kernel suite are on the per-row path today. (The other
  // half of this trade -- StructuredToMemref's scf.for + scf.if lowering --
  // went with that pass on 2026-09-03 and is no longer a cost.)
  let arguments = (ins TT_PtrLike:$ptr,
                       Variadic<Index>:$mask_dims,
                       // THE LOW IS A BOX END LIKE THE DIM, so it is static OR
                       // dynamic like one. A padded convolution's lower bound
                       // is `i - PAD + h`, which is a value and not a number.
                       Variadic<Index>:$mask_lows,
                       DenseI64ArrayAttr:$static_mask_dims,
                       OptionalAttr<DenseI64ArrayAttr>:$static_mask_lows,
                       Optional<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr]>>:$other,
                       Optional<TT_BoolLike>:$elem_mask);

  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "ArrayRef<OpFoldResult>":$mask_dims,
                 "ArrayRef<OpFoldResult>":$mask_lows, "Value":$other,
                 CArg<"Value", "Value()">:$elem_mask,
                 CArg<"Attribute", "Attribute()">:$encoding)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getMixedMaskDims().empty();
    }

    /// The first VALID position on each axis, all zeros when nothing says
    /// otherwise: axis i is live on [lows[i], lows[i] + mask_dims[i]). Absent is
    /// what a mask with no lower bound leaves, and what every older reader
    /// assumed, so a consumer that cannot honour a low must ask hasMaskLow().
    SmallVector<OpFoldResult> getMixedMaskLows() {
      Builder b(getContext());
      if (!getStaticMaskLows())
        return SmallVector<OpFoldResult>(getStaticMaskDims().size(),
                                         b.getIndexAttr(0));
      return ::mlir::getMixedValues(*getStaticMaskLows(), getMaskLows(), b);
    }

    bool hasMaskLow() {
      if (!getMaskLows().empty())
        return true;
      auto lows = getStaticMaskLows();
      return lows && llvm::any_of(*lows, [](int64_t v) { return v != 0; });
    }
  }];

  // The box's three lengths have to agree, and every consumer was checking it.
  let hasVerifier = 1;
  // TODO
  //let hasCustomAssemblyFormat = 1;
}

def TTS_StoreOp : TTS_Op<"store", [
  MemoryEffects<[MemWrite]>,
  AttrSizedOperandSegments,
  OptionalTypesMatchWith<"elem_mask is an i1 of the value's shape", "value",
                         "elem_mask", "triton::getI1SameShape($_self)">
]> {
  let summary = "optionally store data from in memory to fill a portion of the tensor";

  // elem_mask: see tts.load. AttrSizedOperandSegments comes with it -- a
  // variadic `mask_dims` and an optional operand cannot be told apart by
  // position, and this op had no segments before.
  let arguments = (ins TT_PtrLike:$ptr,
                       TT_Tensor:$value,
                       Variadic<Index>:$mask_dims,
                       // See tts.load's: a low is a box end and may be a value.
                       Variadic<Index>:$mask_lows,
                       DenseI64ArrayAttr:$static_mask_dims,
                       OptionalAttr<DenseI64ArrayAttr>:$static_mask_lows,
                       Optional<TT_BoolLike>:$elem_mask);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "Value":$value, "ArrayRef<OpFoldResult>":$dims,
                 "ArrayRef<OpFoldResult>":$lows,
                 CArg<"Value", "Value()">:$elem_mask)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getMixedMaskDims().empty();
    }

    /// The first VALID position on each axis, all zeros when nothing says
    /// otherwise: axis i is live on [lows[i], lows[i] + mask_dims[i]). Absent is
    /// what a mask with no lower bound leaves, and what every older reader
    /// assumed, so a consumer that cannot honour a low must ask hasMaskLow().
    SmallVector<OpFoldResult> getMixedMaskLows() {
      Builder b(getContext());
      if (!getStaticMaskLows())
        return SmallVector<OpFoldResult>(getStaticMaskDims().size(),
                                         b.getIndexAttr(0));
      return ::mlir::getMixedValues(*getStaticMaskLows(), getMaskLows(), b);
    }

    bool hasMaskLow() {
      if (!getMaskLows().empty())
        return true;
      auto lows = getStaticMaskLows();
      return lows && llvm::any_of(*lows, [](int64_t v) { return v != 0; });
    }
  }];

  // The box's three lengths have to agree, and every consumer was checking it.
  let hasVerifier = 1;
  // TODO
  //let hasCustomAssemblyFormat = 1;
}

#endif // TRITON_STRUCTURED_DIALECT
