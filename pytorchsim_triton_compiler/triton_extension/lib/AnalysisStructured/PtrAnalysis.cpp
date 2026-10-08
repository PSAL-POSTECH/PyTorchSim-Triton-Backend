//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include "TorchsimExt/AnalysisStructured/PtrAnalysis.h"

#include "llvm/ADT/ScopeExit.h"
#include "TorchsimExt/Utils/Encoding.h"
#include "TorchsimExt/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "TorchsimExt/Analysis/MaskAnalysis.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "TorchsimExt/Analysis/OpFoldResultUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "TorchsimExt/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <cstddef>
#include <optional>
#include <queue>
#include <string>

#define DEBUG_TYPE "triton-ptr-analysis"

using namespace mlir;

// Try to apply unstructured mask on the ptr.
// A wrapped make_tptr is refused: the gather op has no `shape` field. The
// division below is safe because a base offset never sits on a stride-0 axis.
// A MASK ON AN AXIS OF ONE POSITION IS A SCALAR PREDICATE. It cannot vary along
// an axis that has nowhere to vary -- `um[:, None]` broadcast over a row gives
// every element the same bool -- so it is that axis's EXTENT, 1 or 0, and needs
// neither a gather nor a per-element slot. Runs BEFORE either, or the cheapest
// answer is the one nobody reaches.
static void foldScalarPredicate(triton::MaskState &mstate, Location loc,
                                OpBuilder &builder) {
  auto masks = mstate.getUnstructuredMasks();
  if (masks.size() != 1) {
    return;
  }
  auto [dim, mask] = masks[0];
  auto one = getConstantIntValue(mstate.dims[dim]);
  if (!one || *one != 1) {
    return;
  }
  auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
  if (!maskTy || maskTy.getNumElements() != 1) {
    return;
  }
  SmallVector<Value> zeros(
      maskTy.getRank(),
      arith::ConstantIndexOp::create(builder, loc, 0).getResult());
  Value pred =
      tensor::ExtractOp::create(builder, loc, mask, zeros).getResult();
  Value kept = arith::ConstantIndexOp::create(builder, loc, 1).getResult();
  Value gone = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mstate.dims[dim] =
      arith::SelectOp::create(builder, loc, pred, kept, gone).getResult();
  mstate.masks[dim] = nullptr;
}

// `index` HAS NO WIDTH IN THE TYPE SYSTEM and 64 is the only one it has here;
// getIntOrFloatBitWidth() aborts rather than say so.
static unsigned intWidthOf(Type t) {
  if (auto shaped = dyn_cast<ShapedType>(t))
    t = shaped.getElementType();
  return isa<IndexType>(t) ? 64u : t.getIntOrFloatBitWidth();
}

// DOES THIS CAST CARRY THE VALUE THROUGH UNCHANGED? Only a sign extension
// does: an unsigned one adds 2^N to a negative input and a truncation drops
// however many 2^N it must. `std::nullopt` when the op is not a width change.
//
// ASKED BY DIRECTION, NOT BY NAME, because `arith.index_cast` is both -- it
// widens an i32 to `index` and NARROWS an `index` to i32, and the corpus has
// the second (addptr_scalar_for.mlir and three more).
static std::optional<bool> widthChangeKeepsValue(Operation *def) {
  if (!isa<arith::ExtSIOp, arith::ExtUIOp, arith::IndexCastOp,
           arith::IndexCastUIOp, arith::TruncIOp>(def))
    return std::nullopt;
  bool widens = intWidthOf(def->getResult(0).getType()) >=
                intWidthOf(def->getOperand(0).getType());
  return widens && isa<arith::ExtSIOp, arith::IndexCastOp>(def);
}

static Value rebuildGatherWithMask(Operation *op, Value ptr,
                                   triton::MaskState &mstate, Location loc,
                                   OpBuilder builder) {
  // WILL THE POINTER TAKE THIS MASK? It will when the pointer is already a
  // gather and the mask is on the gathered axis -- that mask says which rows
  // participate, which is what a partial gather is. Then it rebuilds the op
  // with the mask in the slot upstream gave it, dropping nothing: a gather
  // pointer has no shape or order to drop and the rung does not change.
  // A null RESULT IS NOT A FAILURE, it is "the pointer cannot say this" --
  // both callers then put the mask on the access instead.
  SmallVector<std::pair<unsigned, Value>> masks = mstate.getUnstructuredMasks();
  if (masks.empty()) {
    return ptr;
  }
  if (masks.size() > 1) {
    LLVM_DEBUG(op->emitRemark(
        "MaskAnalysis failed for more than one unstructured masks"));
    return nullptr;
  }

  auto [dim, unstructuredMask] = masks[0];

  auto gatherScatterPtr = ptr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>();
  if (!gatherScatterPtr) {
    LLVM_DEBUG(op->emitRemark("PtrAnalysis: not a gather pointer, so the mask "
                              "goes on the access"));
    return nullptr;
  }
  if (dim != gatherScatterPtr.getGatherScatterDim()) {
    LLVM_DEBUG(op->emitRemark("PtrAnalysis: the mask is not on the gathered "
                              "axis, so the mask goes on the access"));
    return nullptr;
  }

  ptr = tts::MakeGatherScatterTensorPtrOp::create(
            builder, loc, gatherScatterPtr.getBase(),
            gatherScatterPtr.getGatherScatterOffset(), unstructuredMask,
            gatherScatterPtr.getGatherScatterDim(),
            gatherScatterPtr.getSizes(), gatherScatterPtr.getMixedStrides(),
            gatherScatterPtr.getMixedOffsets(),
            gatherScatterPtr.getMixedShape(),
            gatherScatterPtr.getBaseOffset())
            .getResult();
  // Clear the mask size for gather/scatter dim. The 0 is a FLAG -- "the box
  // gave up on this axis" -- so the axis has no box left to have a low end of
  // either, and leaving one would name a bound on an extent that is not one.
  mstate.dims[dim] = OpFoldResult(builder.getI32IntegerAttr(0));
  if (mstate.hasLows())
    mstate.lows[dim] = builder.getIndexAttr(0);
  return ptr;
}

namespace mlir {

namespace tts {

SmallVector<OpFoldResult> PtrState::offsetsOf() const {
  SmallVector<OpFoldResult> v;
  for (auto &a : axes)
    v.push_back(a.offset);
  return v;
}
SmallVector<OpFoldResult> PtrState::sizesOf() const {
  SmallVector<OpFoldResult> v;
  for (auto &a : axes)
    v.push_back(a.size);
  return v;
}
SmallVector<OpFoldResult> PtrState::stridesOf() const {
  SmallVector<OpFoldResult> v;
  for (auto &a : axes)
    v.push_back(a.stride);
  return v;
}
SmallVector<OpFoldResult> PtrState::shapeOf() const {
  SmallVector<OpFoldResult> v;
  for (auto &a : axes)
    v.push_back(a.shape);
  return v;
}
SmallVector<int64_t> PtrState::stepsOf() const {
  SmallVector<int64_t> v;
  for (auto &a : axes)
    v.push_back(a.step);
  return v;
}
void PtrState::setAxes(ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes,
                       ArrayRef<OpFoldResult> strides,
                       ArrayRef<OpFoldResult> shape) {
  axes.clear();
  for (uint64_t i = 0; i < offsets.size(); i++)
    axes.push_back({offsets[i], sizes[i], strides[i], shape[i]});
}

int32_t PtrState::getRank() const {
  // THE FIVE-VECTORS-AGREE ASSERT IS GONE because it cannot be false any more:
  // one axis is one object. What is left is the split, which is still a
  // grouping over these and still only a convention.
  assert((split.empty() ||
          std::accumulate(split.begin(), split.end(), 0) ==
              (int32_t)axes.size()) &&
         "split must account for every state axis");
  return axes.size();
}

int64_t PtrState::stepOf(uint32_t d) const { return axes[d].step; }

bool PtrState::hasStep() const {
  return llvm::any_of(axes, [](const Axis &a) { return a.step != 0; });
}

int32_t PtrState::tensorRank() const {
  return split.empty() ? getRank() : (int32_t)split.size();
}

std::pair<int32_t, int32_t> PtrState::axisRange(int32_t t) const {
  // ONE PAST THE END IS A LEGAL ASK: expand_dims appends there.
  if (t >= tensorRank())
    return {getRank(), getRank()};
  if (split.empty())
    return {t, t + 1};
  int32_t begin = std::accumulate(split.begin(), split.begin() + t, 0);
  return {begin, begin + split[t]};
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar && !baseOffset && !flatShape);
}

bool PtrState::hasModulo() const {
  if (flatShape)
    return true;
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimHasModulo(i)) {
      return true;
    }
  }
  return false;
}

bool PtrState::dimHasModulo(uint32_t dim) const {
  assert(
      !isBlockPtr() &&
      "Analysis should not check modulo if PtrState describes block pointer");

  assert(dim < getRank());

  auto intAttr = getIntAttr(axes[dim].shape);
  if (!intAttr.has_value()) {
    return true;
  }

  return intAttr.value() != 0;
}

bool isNotStructured(OpFoldResult offset) {
  auto value = dyn_cast<Value>(offset);
  return value && isa<ShapedType>(value.getType());
}

bool PtrState::dimIsStructured(uint32_t dim) const {
  assert(dim < getRank());

  return !isNotStructured(axes[dim].offset);
}

int32_t PtrState::getNonStructuredDim() const {
  SmallVector<int32_t> dims;
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimIsStructured(i))
      continue;
    dims.emplace_back(i);
  }
  assert(dims.size() == 1 && "must have single non-continuous dimension");
  return dims.front();
}

bool PtrState::noStructuredDimExists() const {
  return getRank() > 0 && llvm::all_of(axes, [](const Axis &a) {
           return isNotStructured(a.offset);
         });
}

bool PtrState::isStructured() const {
  return llvm::all_of(
      axes, [](const Axis &a) { return !isNotStructured(a.offset); });
}

bool PtrState::isBlockPtr() const { return !order.empty(); }

bool isNotSingleDim(Value v) {
  auto shapedTy = dyn_cast<ShapedType>(v.getType());
  if (!shapedTy)
    return false;
  auto valShape = shapedTy.getShape();

  // Make sure there are more than 1 dimensions with size > 1.
  return llvm::find_singleton<int64_t>(
             valShape,
             [](int64_t size, bool) {
               return size > 1 ? (int64_t *)size : nullptr;
             },
             false) == nullptr;
}

// Does this value depend on something READ FROM MEMORY? Asked of the whole
// definition chain, because a loaded coordinate is scaled, floored and
// converted before it becomes an address -- and only the load says so.
static bool dependsOnMemory(Value v, llvm::SmallPtrSet<Operation *, 16> &seen) {
  Operation *def = v.getDefiningOp();
  // A BLOCK ARGUMENT ENDS THE WALK, and ends it as "no": a function parameter
  // or an iter-arg says nothing about data on its own, and claiming otherwise
  // would move values out of the column a reader is meant to act on.
  if (!def || !seen.insert(def).second)
    return false;
  auto iface = dyn_cast<MemoryEffectOpInterface>(def);
  if (iface && iface.hasEffect<MemoryEffects::Read>())
    return true;
  for (Value operand : def->getOperands())
    if (dependsOnMemory(operand, seen))
      return true;
  return false;
}

// Is this type a pointer, or a tensor of them?
static bool isPointerLike(Type t) {
  if (auto shaped = dyn_cast<ShapedType>(t))
    t = shaped.getElementType();
  return isa<triton::PointerType>(t);
}

// The distinct pointer ROOTS feeding this value -- values with no defining op,
// so a function parameter or an iter-arg. Two of them means two bases.
static void collectPointerRoots(Value v,
                               llvm::SmallPtrSet<Operation *, 16> &seen,
                               llvm::SmallPtrSet<void *, 4> &roots) {
  Operation *def = v.getDefiningOp();
  if (!def) {
    if (isPointerLike(v.getType()))
      roots.insert(v.getAsOpaquePointer());
    return;
  }
  if (!seen.insert(def).second)
    return;
  for (Value operand : def->getOperands())
    collectPointerRoots(operand, seen, roots);
}

LogicalResult PtrAnalysis::recordDemotion(Demoted cause, LogicalResult rebuilt,
                                          Value operand) {
  const bool ok = rebuilt.succeeded();
  // A DECLINE IS THE ACCESS'S PROXIMATE CAUSE, so it names itself here. The
  // addptr parks whatever `lastRefusal` holds, and a decline that leaves it
  // empty makes the access report the mechanical consequence instead -- "the
  // pointer was not replaced with tts.make_tptr", which is what happened and
  // never why.
  if (!ok && operand)
    lastRefusal =
        isNotSingleDim(operand)
            ? "the value varies on more than one axis, and a gather pointer "
              "has one gathered axis"
            : "the value could not be rebuilt as an index table";
  if (operand && operand.getDefiningOp()) {
    auto name = operand.getDefiningOp()->getName().getStringRef().str();
    llvm::SmallPtrSet<Operation *, 16> seen;
    if (dependsOnMemory(operand, seen)) {
      ok ? ++demotionCounts.fromMemory : ++demotionCounts.refusedFromMemory;
      ++(ok ? demotionCounts.memoryOps
            : demotionCounts.refusedMemoryOps)[name];
      return rebuilt;
    }
    llvm::SmallPtrSet<Operation *, 16> walked;
    llvm::SmallPtrSet<void *, 4> roots;
    collectPointerRoots(operand, walked, roots);
    if (roots.size() > 1) {
      ok ? ++demotionCounts.manyBases : ++demotionCounts.refusedManyBases;
      ++(ok ? demotionCounts.baseOps : demotionCounts.refusedBaseOps)[name];
      return rebuilt;
    }
    ++(ok ? demotionCounts.otherOps : demotionCounts.refusedOtherOps)[name];
  }
  switch (cause) {
  case Demoted::Rem:
    ok ? ++demotionCounts.rem : ++demotionCounts.refusedRem;
    break;
  case Demoted::Div:
    ok ? ++demotionCounts.div : ++demotionCounts.refusedDiv;
    break;
  case Demoted::Sub:
    ok ? ++demotionCounts.sub : ++demotionCounts.refusedSub;
    break;
  case Demoted::Chosen:
    ok ? ++demotionCounts.chosen : ++demotionCounts.refusedChosen;
    break;
  case Demoted::Other:
    ok ? ++demotionCounts.other : ++demotionCounts.refusedOther;
    break;
  }
  return rebuilt;
}

LogicalResult PtrState::rebuildAsUnsupportedOp(Value operand) {
  if (isNotSingleDim(operand))
    return failure();

  if (!isEmpty())
    return failure();

  // Scalar has been take care early.
  // Assume here must be shape type.
  auto opType = cast<ShapedType>(operand.getType());
  // Skip support for pointer types which could be source of PtrState.
  // This check avoids creating a PtrState with non-structured source.
  if (isa<triton::PointerType>(opType.getElementType()))
    return failure();

  auto opShape = opType.getShape();

  // Setup state for unsupported operation.
  auto indexTy = IndexType::get(operand.getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  auto index1 = IntegerAttr::get(indexTy, APInt(64, 1));
  for (auto size : opShape) {
    bool unit = size == 1;
    axes.push_back({unit ? index0 : OpFoldResult(operand),
                    IntegerAttr::get(indexTy, APInt(64, size)),
                    unit ? index0 : index1, index0});
  }
  return success();
}

// `op` becomes the gather dim's offset. Nothing else is cleared: the terms that
// used to be said twice -- a splatted scalar, a base parked on a unit dim --
// have a slot of their own now and never reach this state.
LogicalResult PtrState::rebuildAsGatherScatter(Value op, int nonContinuousDim) {
  if (isNotSingleDim(op))
    return failure();
  if (nonContinuousDim >= getRank())
    return failure();

  // Scalar has been take care early.
  // Assume here must be shape type.
  auto opShape = cast<ShapedType>(op.getType()).getShape();
  // Make sure the op only contribute to nonContinuousDim by check
  // nonContinuousDim is the dimension with size > 1.
  if (opShape[nonContinuousDim] <= 1)
    return failure();

  // Setup state for nonContinuousDim.
  auto indexTy = IndexType::get(op.getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  auto index1 = IntegerAttr::get(indexTy, APInt(64, 1));

  axes[nonContinuousDim].offset = op;
  axes[nonContinuousDim].stride = index1;
  axes[nonContinuousDim].shape = index0;

  return success();
}

// Null means zero, so a state that never saw a splat adds nothing.
static OpFoldResult addBaseOffsets(OpFoldResult a, OpFoldResult b, Location loc,
                                   OpBuilder &builder) {
  if (!a)
    return b;
  if (!b)
    return a;
  return addOFRs(a, b, loc, builder);
}

// Where a base offset lands: an axis that does not wrap, preferring one with a
// real stride so a gather rebuild has an index to carry. Offsets are absolute
// and the lowerings SUM them, so any such axis names the same address.
// ONE ANSWER FOR EVERY CALLER THAT HAS TO BUILD A TENSOR OFFSET. Three places
// used to pick a width of their own -- the flatten took it from the tensor the
// term joins, visitOperandExtSI from the extsi's result, the gather rebuild
// wrote i32 -- and nothing made them agree.
Type PtrState::tensorOffsetType(OpBuilder &builder) const {
  Type widest;
  for (auto [ofr, _sz, _st, _sh, _k] : axes) {
    auto v = dyn_cast<Value>(ofr);
    if (!v)
      continue;
    auto t = dyn_cast<RankedTensorType>(v.getType());
    // INTEGERS ONLY, because the next line asks a bit width. `index` has none
    // and getIntOrFloatBitWidth() aborts on it -- widenOffsets below already
    // spells the same test this way.
    if (!t || !isa<IntegerType>(t.getElementType()))
      continue;
    auto e = t.getElementType();
    if (!widest || e.getIntOrFloatBitWidth() > widest.getIntOrFloatBitWidth())
      widest = e;
  }
  // i32 WHEN NOTHING SAYS OTHERWISE, which is what tt.make_range is and what
  // the structured offsets around it index-cast to.
  return widest ? widest : builder.getI32Type();
}

void PtrState::widenOffsets(Type elemTy, OpBuilder &builder, Location loc) {
  if (!elemTy || !isa<IntegerType>(elemTy))
    return;
  for (auto &a : axes) {
    auto &ofr = a.offset;
    auto v = dyn_cast<Value>(ofr);
    if (!v)
      continue;
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getElementType() == elemTy ||
        !isa<IntegerType>(t.getElementType()) ||
        t.getElementType().getIntOrFloatBitWidth() >=
            elemTy.getIntOrFloatBitWidth())
      continue;
    ofr = arith::ExtSIOp::create(builder, loc, t.clone(elemTy), v).getResult();
  }
}

// The implicit index of a structured axis, MATERIALISED AT THE NEIGHBOUR'S OWN
// RANK: a range along `axis`, raised to that rank and broadcast. It used to be
// a bare rank-1 range, so a `512x1` neighbour -- the same index, one unit axis
// later -- had no answer and the merge failed. It stands in the neighbour's
// layout `enc`: the range in its slice, raised back one axis at a time.
static Value indexAlongAxis(OpBuilder &builder, Location loc,
                            ArrayRef<int64_t> tileShape, uint64_t axis,
                            Type elemTy, Attribute enc = nullptr) {
  Attribute rowEnc = triton::axisEncoding(enc, tileShape.size(), axis);
  if (!rowEnc)
    enc = nullptr;
  auto rowTy = RankedTensorType::get({tileShape[axis]}, elemTy, rowEnc);
  auto i32RowTy =
      RankedTensorType::get({tileShape[axis]}, builder.getI32Type(), rowEnc);
  Value v =
      triton::MakeRangeOp::create(builder, loc, i32RowTy, 0, tileShape[axis])
          .getResult();
  if (elemTy != builder.getI32Type())
    v = arith::ExtSIOp::create(builder, loc, rowTy, v).getResult();
  // Insert the unit axes in increasing order: every position below `d` is
  // already placed by then, so `d` is also the insertion index.
  for (uint64_t d = 0; d < tileShape.size(); d++)
    if (d != axis)
      v = triton::ExpandDimsOp::create(builder, loc, v, d).getResult();
  auto tileTy = RankedTensorType::get(tileShape, elemTy, enc);
  if (tileTy != v.getType())
    v = triton::BroadcastOp::create(builder, loc, tileTy, v).getResult();
  return v;
}

// Merge two states per dim. A wrap forces a base offset onto an axis that does
// not wrap; a mixed dim is flattened, materialising the structured side's
// implicit index; and rhs may add nothing to a wrapped dim, stride included.
// SPLITTING IS NOT A LOCAL DECISION. Two states that meet at an add describe
// the same TENSOR, so where one has split an axis and the other has not, the
// coarser side is split to match -- `p*s == outer*(K*s) + inner*s` is an
// identity whenever K divides the extent, so the alignment costs nothing.
// Returns failure when the two splits of one axis have no common refinement.
// DOES `k` DIVIDE THIS VALUE, whatever it turns out to be at runtime. A base
// offset is usually a loop's own offset, and a loop whose lower bound and step
// both divide is one whose every induction value does -- so the question the
// division has to ask is divisibility, not constancy.
static bool dividesValue(Value v, int64_t k, unsigned depth = 0) {
  if (k <= 1)
    return true;
  if (auto c = getConstantIntValue(v))
    return *c % k == 0;
  if (depth > 8 || !v)
    return false;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto forOp = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
    // THE INDUCTION VARIABLE ONLY. An iter-arg is whatever the body yields and
    // this walk does not follow a back edge.
    if (!forOp || arg != forOp.getInductionVar())
      return false;
    return dividesValue(forOp.getLowerBound(), k, depth + 1) &&
           dividesValue(forOp.getStep(), k, depth + 1);
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  // A PRODUCT NEEDS ONE FACTOR, A SUM NEEDS BOTH, and only a cast that keeps
  // the value carries divisibility: one that moves it by 2^N moves it off any
  // multiple of a k that does not divide 2^N.
  if (isa<arith::MulIOp>(def))
    return dividesValue(def->getOperand(0), k, depth + 1) ||
           dividesValue(def->getOperand(1), k, depth + 1);
  if (isa<arith::AddIOp, arith::SubIOp>(def))
    return dividesValue(def->getOperand(0), k, depth + 1) &&
           dividesValue(def->getOperand(1), k, depth + 1);
  if (std::optional<bool> keeps = widthChangeKeepsValue(def))
    return *keeps && dividesValue(def->getOperand(0), k, depth + 1);
  return false;
}

static bool dividesOFR(OpFoldResult ofr, int64_t k) {
  if (auto c = getConstantIntValue(ofr))
    return *c % k == 0;
  auto v = dyn_cast<Value>(ofr);
  return v && dividesValue(v, k);
}

// THE LARGEST NUMBER THIS VALUE CANNOT GO BELOW, structurally. Reaching the
// depth guard answers "unknown", never a bound. `isNonNegative` is this asked
// at zero, and that question is the whole soundness argument for reading a bit
// mask as a wrap -- `-3 & 7` is 5 and `-3 rem 8` is -3.
static std::optional<int64_t> lowerBoundOf(Value v, unsigned depth = 8) {
  if (!v || depth == 0)
    return std::nullopt;
  if (auto c = triton::constantIntOrSplat(v))
    return c;
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  auto of = [&](unsigned i) { return lowerBoundOf(def->getOperand(i), depth - 1); };
  // A LANE INDEX AND A PROGRAM ID SAY IT THEMSELVES.
  if (auto range = dyn_cast<triton::MakeRangeOp>(def))
    return int64_t(range.getStart());
  if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(def))
    return int64_t(0);
  // A MASK BY A NON-NEGATIVE CLEARS THE SIGN BIT whatever the other side is,
  // and a maximum is at least the larger of the two bounds it has.
  if (isa<arith::AndIOp>(def)) {
    auto a = of(0), b = of(1);
    if ((a && *a >= 0) || (b && *b >= 0))
      return int64_t(0);
    return std::nullopt;
  }
  if (isa<arith::MaxSIOp>(def)) {
    auto a = of(0), b = of(1);
    if (a && b)
      return std::max(*a, *b);
    return a ? a : b;
  }
  if (isa<arith::MinSIOp>(def)) {
    auto a = of(0), b = of(1);
    return (a && b) ? std::optional<int64_t>(std::min(*a, *b)) : std::nullopt;
  }
  // A SUM ADDS ITS BOUNDS. A PRODUCT ONLY WHEN BOTH SIDES ARE NON-NEGATIVE,
  // because a negative factor turns the smaller bound into the larger product.
  if (isa<arith::AddIOp>(def)) {
    auto a = of(0), b = of(1);
    if (!a || !b)
      return std::nullopt;
    int64_t sum;
    return __builtin_add_overflow(*a, *b, &sum) ? std::nullopt
                                                : std::optional<int64_t>(sum);
  }
  if (isa<arith::MulIOp>(def)) {
    auto a = of(0), b = of(1);
    if (!a || !b || *a < 0 || *b < 0)
      return std::nullopt;
    int64_t prod;
    return __builtin_mul_overflow(*a, *b, &prod) ? std::nullopt
                                                 : std::optional<int64_t>(prod);
  }
  // A REMAINDER AND A QUOTIENT OF NON-NEGATIVES ARE NON-NEGATIVE, and 0 is all
  // this needs to say about them.
  if (isa<arith::RemSIOp, arith::RemUIOp, arith::DivSIOp, arith::DivUIOp>(def)) {
    auto a = of(0), b = of(1);
    return (a && *a >= 0 && b && *b >= 0) ? std::optional<int64_t>(0)
                                          : std::nullopt;
  }
  // A WIDTH CHANGE IS NOT ALWAYS A NO-OP. A sign extension carries the bound;
  // an UNSIGNED widening lands in [0, 2^N) so 0 is a bound even where the
  // input has none; a NARROWING keeps no bound at all, because the value it
  // drops 2^N from can land anywhere.
  if (std::optional<bool> keeps = widthChangeKeepsValue(def)) {
    if (*keeps)
      return of(0);
    if (intWidthOf(def->getResult(0).getType()) <=
        intWidthOf(def->getOperand(0).getType()))
      return std::nullopt;
    auto lo = of(0);
    return (lo && *lo > 0) ? lo : std::optional<int64_t>(0);
  }
  // AND THESE ONLY MOVE THE VALUE'S SHAPE.
  if (isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
          triton::ReshapeOp>(def))
    return of(0);
  return std::nullopt;
}

static bool isNonNegative(Value v) {
  auto lo = lowerBoundOf(v);
  return lo && *lo >= 0;
}


// EVERY ELEMENT OF THIS TERM IS >= 0, or we do not know. An absent term adds
// nothing, which is the one case that needs no proof.
static bool nonNegativeOFR(OpFoldResult ofr) {
  if (!ofr)
    return true;
  if (auto c = getConstantIntValue(ofr))
    return *c >= 0;
  auto v = dyn_cast<Value>(ofr);
  return v && isNonNegative(v);
}

// `v / k` WHERE THE DIVISION IS KNOWN EXACT -- dividesOFR said so, so nothing
// is lost and no rounding rule is being chosen.
static OpFoldResult divideExactly(OpFoldResult ofr, int64_t k,
                                  const Location loc, OpBuilder &builder) {
  if (auto c = getConstantIntValue(ofr))
    return builder.getIndexAttr(*c / k);
  auto v = cast<Value>(ofr);
  auto d = arith::ConstantOp::create(builder, loc,
                                     builder.getIntegerAttr(v.getType(), k));
  return arith::DivSIOp::create(builder, loc, v, d).getResult();
}

// ONE PLACE, ONE STRING -- see the declaration. The remark says who is
// speaking; `why` is what report-demotions counts, and there is now only one of
// it.
LogicalResult PtrAnalysis::refuse(Operation *op, const char *why) {
  LLVM_DEBUG(op->emitRemark() << "PtrAnalysis: " << why);
  lastRefusal = why;
  return failure();
}

// The same, for the PtrState methods -- they cannot reach the analysis, so the
// name travels out through the Refusal the caller passed in.
static LogicalResult refuse(Operation *op, Refusal *refusal, const char *why,
                            bool fromStep = false) {
  LLVM_DEBUG(op->emitRemark() << "PtrAnalysis: " << why);
  if (refusal)
    *refusal = {why, fromStep};
  return failure();
}

// CUT ONE STATE AXIS IN TWO. The ONLY way a state grows an axis without the
// tensor growing one, and the only place `split` is bumped -- which is what
// tells this apart from expandTensorAxis below. The caller has already written
// the outer half into `axes[d]`; this appends the inner one.
// GIVE THE TENSOR A NEW AXIS, and give the state the one state axis that
// carries it. The counterpart of cutStateAxis: that one leaves the tensor's
// rank alone and bumps `split`, this one raises both by one. Naming them apart
// is the whole difference -- both used to be an insert into `sizes`.
static void expandTensorAxis(PtrState &st, int32_t t, uint32_t at,
                             PtrState::Axis a) {
  st.axes.insert(st.axes.begin() + at, a);
  if (!st.split.empty())
    st.split.insert(st.split.begin() + t, 1);
}

// AND TAKE ONE BACK. A widened axis is stride 0 throughout, which makes its
// split meaningless, so the group collapses to the one axis that is left.
static void collapseTensorAxisGroup(PtrState &st, int32_t t, uint32_t b,
                                    uint32_t e) {
  st.axes.erase(st.axes.begin() + b + 1, st.axes.begin() + e);
  st.split[t] = 1;
}

static void cutStateAxis(PtrState &st, int32_t t, uint32_t d,
                         PtrState::Axis inner) {
  if (st.split.empty())
    st.split.assign(st.tensorRank(), 1);
  st.axes.insert(st.axes.begin() + d + 1, inner);
  st.split[t] += 1;
}

static LogicalResult splitAxisBy(PtrState &st, int32_t t, int64_t k,
                                 int64_t outerStride, int64_t innerStride,
                                 OpBuilder &builder) {
  auto [b, e] = st.axisRange(t);
  assert(e == b + 1 && "only an unsplit axis is split here");
  auto n = getConstantIntValue(st.axes[b].size);
  if (!n || k <= 1 || *n % k != 0 || !hasConstZero(st.axes[b].shape) ||
      st.stepOf(b))
    return failure();
  st.axes[b].size = builder.getIndexAttr(*n / k);
  st.axes[b].stride = builder.getIndexAttr(outerStride);
  cutStateAxis(st, t, b,
               {builder.getIndexAttr(0), builder.getIndexAttr(k),
                builder.getIndexAttr(innerStride), builder.getIndexAttr(0)});
  return success();
}

// CUT ONE STATE AXIS OF A GROUP THAT MAY ALREADY BE SPLIT. splitAxisBy above
// takes a tensor axis and refuses one that has been cut; this takes the state
// axis, so a group can be refined again. `k` is the INNER extent either way.
//
// `p*s == outer*(k*s) + inner*s` is an identity, so a plain axis costs nothing.
// A WRAPPED ONE SURVIVES ONLY WHEN THE WRAP IS THE NEW OUTER STRIDE: at W ==
// s*k the outer term is a whole number of laps and vanishes, `(o*W + i*s) mod W
// == i*s`, so the wrap goes with it. Any other W mixes the two halves and no
// pair of per-axis wraps says it.
static LogicalResult refineStateAxis(PtrState &st, int32_t t, uint32_t d,
                                     int64_t k, OpBuilder &builder,
                                     Refusal *refusal = nullptr) {
  auto n = getConstantIntValue(st.axes[d].size);
  auto s = getConstantIntValue(st.axes[d].stride);
  // A STEPPED AXIS IS NOT CUT, AND THE CALLER HAS TO HEAR WHY. `p // m` over
  // runs of m does not separate into `q` and `r` terms when p is split at k,
  // so the cut has no answer here. Saying so is not cosmetic: `fromStep` is
  // what un-mints the step and visits again, and without it this address falls
  // TWO rungs where every other step refusal costs one.
  if (st.stepOf(d)) {
    if (refusal)
      *refusal = {"one axis cannot both step and split", true};
    return failure();
  }
  if (!n || !s || k <= 1 || *n % k != 0)
    return failure();
  int64_t outerStride = *s * k;
  OpFoldResult innerOffset = builder.getIndexAttr(0);
  OpFoldResult innerShape = builder.getIndexAttr(0);
  if (!hasConstZero(st.axes[d].shape)) {
    auto w = getConstantIntValue(st.axes[d].shape);
    if (!w || *w != outerStride)
      return failure();
    // THE WRAP AND THE OFFSET BOTH RIDE THE INNER HALF. `(off + o*W + i*s) mod
    // W
    // == (off + i*s) mod W`: the outer term is whole laps and vanishes, and
    // what is left is the inner one -- offset included, because it sits INSIDE
    // the modulo and moving it to the outer would take it back out.
    outerStride = 0;
    innerOffset = st.axes[d].offset;
    innerShape = st.axes[d].shape;
    // AND A WRAP THE INNER HALF CANNOT REACH IS NOT A WRAP. The cut can leave
    // the walk short of the boundary -- offset 0 over `[0, s, .., (k-1)*s]`
    // stops one stride below W -- and make_tptr's verifier refuses a fact that
    // is not one.
    if (triton::wrapNeverFires(innerShape, innerOffset,
                               builder.getIndexAttr(*s), k))
      innerShape = builder.getIndexAttr(0);
    st.axes[d].offset = builder.getIndexAttr(0);
    st.axes[d].shape = builder.getIndexAttr(0);
  }
  st.axes[d].size = builder.getIndexAttr(*n / k);
  st.axes[d].stride = builder.getIndexAttr(outerStride);
  cutStateAxis(st, t, d,
               {innerOffset, builder.getIndexAttr(k),
                builder.getIndexAttr(*s), innerShape});
  return success();
}

// THE COARSEST FACTORISATION BOTH REFINE TO, as extents. A factorisation's cut
// points are its suffix products, and two of them meet only when the union of
// those is a single divisibility chain -- [2,3] against [3,2] has no common
// refinement and this is where that is noticed.
static std::optional<SmallVector<int64_t>>
commonRefinement(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  SmallVector<int64_t> cuts;
  auto suffixes = [&](ArrayRef<int64_t> f) {
    int64_t p = 1;
    for (int i = f.size() - 1; i > 0; i--) {
      p *= f[i];
      cuts.push_back(p);
    }
  };
  suffixes(a);
  suffixes(b);
  llvm::sort(cuts, std::greater<int64_t>());
  cuts.erase(llvm::unique(cuts), cuts.end());
  int64_t total = 1;
  for (auto x : a)
    total *= x;
  SmallVector<int64_t> sizes;
  int64_t prev = total;
  for (auto c : cuts) {
    if (c <= 1 || c >= prev || prev % c != 0)
      return std::nullopt;
    sizes.push_back(prev / c);
    prev = c;
  }
  sizes.push_back(prev);
  return sizes;
}

// Cut one tensor axis's group down to `target`, one state axis at a time.
static LogicalResult refineTo(PtrState &st, int32_t t,
                              ArrayRef<int64_t> target, OpBuilder &builder,
                              Refusal *refusal = nullptr) {
  uint32_t d = st.axisRange(t).first;
  for (size_t j = 0; j < target.size(); j++, d++) {
    if (d >= st.axisRange(t).second)
      return failure();
    auto n = getConstantIntValue(st.axes[d].size);
    if (!n)
      return failure();
    if (*n == target[j])
      continue;
    if (*n % target[j] != 0 ||
        refineStateAxis(st, t, d, *n / target[j], builder, refusal).failed())
      return failure();
  }
  return success();
}

// MAKE THE TWO SIDES AGREE ON WHERE THE AXIS IS CUT, not merely on how many
// times. addState reads state axis i of both as one axis, so [4, 64] against
// [128, 2] used to merge and keep the left side's extents -- a wrong answer
// rather than a refusal. Both are refined to their common refinement instead.
static LogicalResult alignSplits(PtrState &a, PtrState &b, OpBuilder &builder,
                                 Refusal *refusal = nullptr) {
  if (a.tensorRank() != b.tensorRank())
    return failure();
  for (int32_t t = 0; t < a.tensorRank(); t++) {
    SmallVector<int64_t> ea, eb;
    auto extents = [&](const PtrState &st, SmallVector<int64_t> &out) {
      auto [x, y] = st.axisRange(t);
      for (auto i = x; i < y; i++) {
        auto n = getConstantIntValue(st.axes[i].size);
        if (!n)
          return false;
        out.push_back(*n);
      }
      return true;
    };
    if (!extents(a, ea) || !extents(b, eb))
      return failure();
    if (ea == eb)
      continue;
    auto target = commonRefinement(ea, eb);
    if (!target || refineTo(a, t, *target, builder, refusal).failed() ||
        refineTo(b, t, *target, builder, refusal).failed())
      return failure();
  }
  return success();
}

// BOTH SIDES DESCRIBING ONE AXIS THE SAME WAY is the precondition every merge
// has, and it is the same question for a sum and a product. The aligned copies
// go out through `lhs`/`rhs`, so the bodies read `lhs.axes[i].stride` and mean
// "state axis i of BOTH sides" -- which is what keeps the split invisible to
// them. The result's own `split` and `step` are filled in here too.
//
// mulState reaches the last guard with one side a scalar, so it cannot fire
// there; carrying it anyway is what makes this ONE question rather than two
// that drift.
// THE MERGED AXES, built column by column and assembled once. addState and
// mulState fill offset and stride in one pass over the axes and the wrap in a
// second, which is why these arrive as four lists rather than as axes. The step
// is not merged at all -- it is read off whichever side had one, the pair
// having already been checked to agree by alignForMerge.
static void assembleAxes(PtrState &out, const PtrState &lhs,
                         const PtrState &rhs, ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes,
                         ArrayRef<OpFoldResult> strides,
                         ArrayRef<OpFoldResult> shape) {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
         offsets.size() == shape.size());
  out.axes.clear();
  for (uint64_t i = 0; i < offsets.size(); i++) {
    int64_t step = 0;
    if (i < (uint64_t)lhs.getRank())
      step = lhs.stepOf(i) ? lhs.stepOf(i) : rhs.stepOf(i);
    out.axes.push_back({offsets[i], sizes[i], strides[i], shape[i], step});
  }
}

LogicalResult PtrState::alignForMerge(const PtrState &lhsIn,
                                      const PtrState &rhsIn, PtrState &lhs,
                                      PtrState &rhs, Operation *op,
                                      OpBuilder &builder, Refusal *refusal) {
  lhs = lhsIn;
  rhs = rhsIn;
  Refusal splitWhy;
  if (alignSplits(lhs, rhs, builder, &splitWhy).failed()) {
    return refuse(op, refusal,
                  splitWhy.why ? splitWhy.why
                               : "the two sides split an axis differently",
                  splitWhy.fromStep);
  }
  split = lhs.split;
  // A STEP IS AN AXIS FACT AND BOTH SIDES DESCRIBE THE SAME AXES, so two
  // different steps on one axis have no single index to offer.
  for (int32_t d = 0; d < lhs.getRank(); d++) {
    if (lhs.stepOf(d) && rhs.stepOf(d) &&
        lhs.stepOf(d) != rhs.stepOf(d)) {
      return refuse(op, refusal,
                    "the two sides step one axis by different periods", true);
    }
    // AND A STEP MEETING A WRAP IS THE SAME REFUSAL, one operand later. `x % W`
    // and `x // K` off ONE value land in different terms of the same address,
    // so the two arrive here rather than at the division -- which is where the
    // guard used to be, and only there.
    if ((lhs.stepOf(d) && !hasConstZero(rhs.axes[d].shape)) ||
        (rhs.stepOf(d) && !hasConstZero(lhs.axes[d].shape))) {
      return refuse(op, refusal, "one axis cannot both step and wrap", true);
    }
    // AND IT CANNOT STEP AND GATHER. The other half of `x % W` reaches the
    // address as an unstructured term on the same axis; two index functions,
    // one axis.
    if ((lhs.stepOf(d) && !rhs.dimIsStructured(d)) ||
        (rhs.stepOf(d) && !lhs.dimIsStructured(d))) {
      return refuse(op, refusal, "one axis cannot both step and gather", true);
    }
    // AND IT CANNOT STEP AND RUN. `p // K` and a plain `p` are two index
    // functions again -- which is how `p % K` arrives once stage 2 has
    // rewritten it to `p` -- and `(p//K)*a + p*b` is not `(p//K)*(a+b)`.
    if ((lhs.stepOf(d) && !rhs.stepOf(d) &&
         !hasConstZero(rhs.axes[d].stride)) ||
        (rhs.stepOf(d) && !lhs.stepOf(d) &&
         !hasConstZero(lhs.axes[d].stride))) {
      return refuse(op, refusal, "one axis cannot both step and run", true);
    }
  }
  // THE STEP IS NOT MERGED HERE. `axes` is still empty at this point -- the
  // caller fills it column by column -- so whichever side carries the step is
  // read back at assembly, which is also where the axis it belongs to exists.
  return success();
}

// ONE AXIS WHERE ONE SIDE'S INDEX IS DATA. A structured axis is `offset +
// p*stride`; an unstructured one is a tensor of indices, and the two do not add
// term by term. This is the whole of how they meet -- which side supplies the
// axis, when an implicit index has to be materialised so it does not fold away,
// and what width the merged index is built at.
// BOTH SIDES HAVE SOMETHING ON THIS AXIS AND ONE OF THEM IS DATA, so there is
// no pair of (offset, stride) that says the sum. The way out is to fold the
// whole term into ONE index tensor and leave the stride at 1: the offset
// becomes `offset * stride` and the address is read off the index. Two
// different reasons arrive here -- an implicit index that has to be
// materialised, and upstream's own case of two differing strides -- and the
// block says which is which where they part.
LogicalResult PtrState::foldDimOntoOneIndex(
    const PtrState &lhsState, const PtrState &rhsState,
    ArrayRef<OpFoldResult> lhsOffsets, uint64_t i,
    SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &strides, Operation *op, OpBuilder &builder,
    Refusal *refusal) {
  auto loc = op->getLoc();
  OpFoldResult lhsOffset = lhsOffsets[i];
  OpFoldResult rhsOffset = rhsState.axes[i].offset;
  OpFoldResult lhsStride = lhsState.axes[i].stride;
  OpFoldResult rhsStride = rhsState.axes[i].stride;
  bool lhsNoIndex = hasConstZero(lhsStride);
  bool rhsNoIndex = hasConstZero(rhsStride);
  // If stride is 0 which will happen after
  // visitOperandExpandDims/visitOperandSplat, we set the stride to 1 to
  // mul it with offset.
  if (hasConstZero(lhsStride)) {
    assert(lhsState.dimIsStructured(i) &&
           !rhsState.dimIsStructured(i) &&
           "If lhs stride is zero, it must be structured and rhs "
           "stride is unstructured");
    lhsStride = builder.getIndexAttr(1);
  }
  if (hasConstZero(rhsStride)) {
    assert(rhsState.dimIsStructured(i) &&
           !lhsState.dimIsStructured(i) &&
           "If rhs stride is zero, it must be structured and lhs "
           "stride is unstructured");
    rhsStride = builder.getIndexAttr(1);
  }

  // If both offset and stride not equal, we merge 2 PtrStates by change
  // offset * stride into (offset * stride) * 1 where new offset is
  // offset * stride and new stride is set to 1.
  // Then we'll have strides equal as 1, and merge them as PtrState with
  // same strides.
  // TWO REASONS REACH THIS BLOCK AND THEY ARE NOT THE SAME QUESTION.
  //   mixedDim  one side's index k is implicit and the other's is
  //             spelled out per element, so the implicit one has to be
  //             materialised or it folds away.
  //   normalise upstream's own reason: the two strides differ and both
  //             are forced to 1. Reachable only with BOTH sides
  //             unstructured, where flatten takes the `offset * stride`
  //             path and does exactly what it always did -- which the
  //             assert below is here to keep true.
  bool mixedDim =
      lhsState.dimIsStructured(i) != rhsState.dimIsStructured(i);
  bool normaliseStrides = lhsOffset != rhsOffset && lhsStride != rhsStride;
  assert((mixedDim || !normaliseStrides ||
          (!lhsState.dimIsStructured(i) &&
           !rhsState.dimIsStructured(i))) &&
         "the stride-normalising arm should only ever see two "
         "unstructured sides");
  bool flattened = false;
  if (mixedDim || normaliseStrides) {
    // THE MERGE'S WIDTH IS THE WIDER OF THE TWO STATES. Asking either
    // side alone gives the wrong answer: the structured side has no
    // tensor offset to read a width off, and it is the side whose
    // implicit index is about to be materialised.
    Type mergedTy = lhsState.tensorOffsetType(builder);
    Type rhsTy = rhsState.tensorOffsetType(builder);
    if (rhsTy.getIntOrFloatBitWidth() > mergedTy.getIntOrFloatBitWidth())
      mergedTy = rhsTy;
    // ONE CAST RULE, AND IT IS NOT OURS TO WRITE. This was spelled
    // out here -- index_cast, extsi, trunci -- and three more times in
    // the repo. It is getValueOrCreateCastToIndexLike, which MLIR
    // already ships, and ofrToValue is the materialising half.
    //
    // THE COPY'S ARMS WERE IN AN ORDER THAT CANNOT BE RIGHT: the
    // index SOURCE test came BEFORE the same-type test, so index to
    // index would emit an invalid arith.index_cast, and an index
    // TARGET fell through to getIntOrFloatBitWidth(), which asserts on
    // IndexType. Neither is reachable, and tensorOffsetType above now
    // skips `index` rather than aborting on it.
    auto toElem = [&](OpFoldResult ofr, Type elemTy) -> Value {
      return getValueOrCreateCastToIndexLike(
          builder, loc, elemTy, ofrToValue(ofr, loc, builder));
    };
    auto flatten = [&](const PtrState &st, OpFoldResult off,
                       OpFoldResult strd, OpFoldResult tyLike,
                       bool noIndex) -> OpFoldResult {
      if (!st.dimIsStructured(i)) {
        OpFoldResult s = expandOFRIndex(strd, off, loc, builder);
        return mulOFRs(off, s, loc, builder);
      }
      auto tyValue = dyn_cast<Value>(tyLike);
      if (!tyValue)
        return mulOFRs(off, strd, loc, builder);
      auto shapedTy = dyn_cast<ShapedType>(tyValue.getType());
      // THE NEIGHBOUR'S RANK IS ITS OWN. A collapsed rank-1 term names
      // this dim with its only axis; a term at the state's rank names
      // it with axis i. Anything else does not say which axis it is.
      if (!shapedTy || (shapedTy.getRank() != 1 &&
                        shapedTy.getRank() != lhsState.getRank()))
        return OpFoldResult();  // caller turns this into a failure
      uint64_t nbAxis = shapedTy.getRank() == 1 ? 0 : i;
      // SHAPE FROM THE NEIGHBOUR, WIDTH FROM THE MERGE. The row has to
      // be as long as the term it joins; how wide it stands is the one
      // answer both sides agreed on above.
      Type elemTy = mergedTy;
      Attribute enc = nullptr;
      if (auto ranked = dyn_cast<RankedTensorType>(shapedTy))
        enc = ranked.getEncoding();
      auto rowTy = RankedTensorType::get(shapedTy.getShape(), elemTy, enc);
      Value base = tensor::SplatOp::create(builder, loc, rowTy,
                                           toElem(off, elemTy))
                       .getResult();
      if (noIndex)
        return OpFoldResult(base);
      Value range = indexAlongAxis(builder, loc, shapedTy.getShape(),
                                   nbAxis, elemTy, enc);
      Value scale = tensor::SplatOp::create(builder, loc, rowTy,
                                            toElem(strd, elemTy))
                        .getResult();
      Value scaled =
          arith::MulIOp::create(builder, loc, range, scale).getResult();
      return OpFoldResult(
          arith::AddIOp::create(builder, loc, base, scaled).getResult());
    };
    OpFoldResult newLhs = flatten(lhsState, lhsOffset, lhsStride,
                                  lhsState.dimIsStructured(i) ? rhsOffset
                                                              : lhsOffset,
                                  lhsNoIndex);
    OpFoldResult newRhs = flatten(rhsState, rhsOffset, rhsStride,
                                  rhsState.dimIsStructured(i) ? lhsOffset
                                                              : rhsOffset,
                                  rhsNoIndex);
    if (!newLhs || !newRhs) {
      return refuse(
          op, refusal,
          "cannot materialise the implicit index of a structured dim "
          "being merged with an unstructured one");
    }
    // A GUARD, NOT A CAPABILITY BOUNDARY. The two halves are built at
    // the rank of the term each JOINED, so they disagree only when this
    // dim carries two spellings of its offset -- and the invariant that
    // says it cannot is kept at the source: an offset stands at the
    // state's rank, so visitOperandExpandDims moves it with the rank.
    // Reaching here means that broke; the refusal lands the access on a
    // gather rather than building `addi(tensor<64xi32>,
    // tensor<64x1xi32>)`, which does not verify and which took
    // resnet18's first conv down for the length of "Refactoring F2".
    {
      auto lhsV = dyn_cast<Value>(newLhs);
      auto rhsV = dyn_cast<Value>(newRhs);
      if (bool(lhsV) != bool(rhsV) ||
          (lhsV && rhsV && lhsV.getType() != rhsV.getType())) {
        return refuse(op, refusal,
                      "the two flattened halves of a merged dim do not "
                      "stand at the same rank");
      }
    }
    lhsOffset = newLhs;
    rhsOffset = newRhs;
    // Set both strides to 1.
    lhsStride = builder.getIndexAttr(1);
    rhsStride = builder.getIndexAttr(1);
    flattened = true;
  }

  if (lhsStride == rhsStride) {
    // For case like lhs_offset * stride + rhs_offset * stride, it is
    // same as (lhs_offset + rhs_offset) * stride. We can just add the
    // offsets and reuse the stride like this:
    //   axes[i].offset = lhsOffset + rhsOffset
    //   axes[i].stride = lhsStride
    // Expand structured offset since unstructured offset has tensor
    // type. Skipped when the dim was flattened above: both sides are
    // already per-element tensors there, and the structured side's
    // index is IN one of them rather than still implicit.
    if (!flattened) {
      // AND THE EXPANSION MUST NOT NARROW. expandOFRIndex gives one side the
      // OTHER side's element type, so an offset standing in `index` could meet
      // an i32 tensor and lose its high bits -- silently, and only on the
      // buffers big enough to have them.
      //
      // AN ASSERT AND NOT AN ARM, because the two sides here already AGREE on
      // structuredness: a dim where they differ went through the flatten above,
      // which takes the WIDER of the two states. Agreeing sides come from one
      // program expression and carry its width, so the narrowing case has no
      // input to demote -- probed 0 over 312 lit files and 462 kernels, and
      // this is what keeps that true.
      [[maybe_unused]] constexpr const char *kNoNarrowing =
          "a merge may not narrow an offset to the other side's width";
      [[maybe_unused]] auto narrows = [&](OpFoldResult src,
                                          OpFoldResult tgt) {
        auto tgtValue = dyn_cast<Value>(tgt);
        if (!tgtValue || !isa<ShapedType>(tgtValue.getType()))
          return false;
        unsigned want = intWidthOf(tgtValue.getType());
        // THE STATE RAISES EVERY INTEGER SCALAR TO `index`, so a type read off
        // the offset is not the width the VALUE needs. Walk back through the
        // casts that kept it and ask the one that minted it.
        auto srcValue = dyn_cast<Value>(src);
        unsigned have = 64u;
        if (srcValue) {
          Value root = srcValue;
          while (Operation *def = root.getDefiningOp()) {
            std::optional<bool> keeps = widthChangeKeepsValue(def);
            if (!keeps || !*keeps)
              break;
            root = def->getOperand(0);
          }
          have = intWidthOf(root.getType());
        }
        if (have <= want)
          return false;
        auto c = getConstantIntValue(src);
        return !c || !APInt(64, *c, /*isSigned=*/true).isSignedIntN(want);
      };
      if (!lhsState.dimIsStructured(i)) {
        assert(!narrows(rhsOffset, lhsOffset) && kNoNarrowing);
        rhsOffset = expandOFRIndex(rhsOffset, lhsOffset, loc, builder);
      } else {
        assert(!narrows(lhsOffset, rhsOffset) && kNoNarrowing);
        lhsOffset = expandOFRIndex(lhsOffset, rhsOffset, loc, builder);
      }
    }
    // Add offsets.
    offsets.push_back(addOFRs(lhsOffset, rhsOffset, loc, builder));
    // Reuse stride.
    strides.push_back(lhsStride);
  } else {
    // Assert that offsets are equal if strides are not equal.
    // This is because we are already forcing the strides to be
    // equal to 1 earlier for case both offsets and strides not equal.
    assert(lhsOffset == rhsOffset &&
           "If strides are not equal, offsets must be equal");
    // For case like offset * lhs_stride + offset * rhs_stride, it is
    // same as offset * (lhs_stride + rhs_stride). We can just add the
    // strides and reuse the offset like this:
    //   axes[i].offset = lhsOffset
    //   axes[i].stride = lhsStride + rhsStride

    // Reuse offsets.
    offsets.push_back(lhsOffset);
    // Add strides.
    strides.push_back(addOFRs(lhsStride, rhsStride, loc, builder));
  }
  return success();
}

LogicalResult PtrState::mergeDimWithUnstructured(
    const PtrState &lhsState, const PtrState &rhsState,
    ArrayRef<OpFoldResult> lhsOffsets, uint64_t i,
    SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &strides, bool isAnalysisingUnstructured,
    Operation *op, OpBuilder &builder, Refusal *refusal) {
  auto loc = op->getLoc();
  if (isAnalysisingUnstructured) {
    if (hasConstZero(lhsState.axes[i].stride) &&
        hasConstZero(lhsOffsets[i])) {
      // If lhs is not for dim i, we can just use rhs's stride and offset.
      offsets.push_back(rhsState.axes[i].offset);
      strides.push_back(rhsState.axes[i].stride);
    } else if (hasConstZero(rhsState.axes[i].stride) &&
               hasConstZero(rhsState.axes[i].offset)) {
      // If rhs is not for dim i, we can just use lhs's stride and offset.
      offsets.push_back(lhsOffsets[i]);
      strides.push_back(lhsState.axes[i].stride);
    } else if (foldDimOntoOneIndex(lhsState, rhsState, lhsOffsets, i, offsets,
                                   strides, op, builder, refusal)
                   .failed()) {
      return failure();
    }
  } else {
    // Set stride to 1 when not continuous.
    strides.push_back(builder.getIndexAttr(1));
    // New offset is offset * stride.
    auto newLhsOffset = lhsOffsets[i];
    auto newRhsOffset = rhsState.axes[i].offset;
    // Just propagate the unstructured offset to the result to track the
    // unstructured dimension. The real address calculation will be done
    // later in the PtrAnalysis::visitOperandAddptr.
    auto newOffset =
        lhsState.dimIsStructured(i) ? newRhsOffset : newLhsOffset;
    offsets.push_back(newOffset);
  }
  return success();
}

LogicalResult PtrState::addState(const PtrState &lhsIn, const PtrState &rhsIn,
                                 bool isAnalysisingUnstructured, Operation *op,
                                 OpBuilder &builder, Refusal *refusal) {
  assert(isEmpty());
  PtrState lhsState, rhsState;
  if (alignForMerge(lhsIn, rhsIn, lhsState, rhsState, op, builder, refusal)
          .failed())
    return failure();
  // BUILT COLUMN BY COLUMN, ASSEMBLED ONCE. The wrap is decided in a second
  // pass over the axes, so the four cannot be filled axis at a time.
  SmallVector<OpFoldResult> offsets, sizes, strides, shape;
  assert(lhsState.getRank() == rhsState.getRank());
  auto loc = op->getLoc();

  if (lhsState.source && rhsState.source) {
    return refuse(op, refusal,
                  "do not support adding two pointer states that both have "
                  "base pointers");
  }

  source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        arith::AddIOp::create(builder, loc, lhsState.scalar, rhsState.scalar);
    scalar = addOp.getResult();
  } else if (lhsState.getRank() == 0) { // both lhs and rhs are scalars
    scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  if (!lhsState.isStructured() && !rhsState.isStructured()) {
    if (lhsState.getNonStructuredDim() != rhsState.getNonStructuredDim()) {
      return refuse(op, refusal,
                    "do not support adding two pointer states that have "
                    "different non-continuous dimension");
    }
  }

  SmallVector<OpFoldResult> lhsOffsets = lhsState.offsetsOf();
  SmallVector<OpFoldResult> rhsOffsets = rhsState.offsetsOf();
  baseOffset = addBaseOffsets(lhsState.baseOffset, rhsState.baseOffset, loc,
                              builder);

  for (uint64_t i = 0; i < lhsState.getRank(); i++) {
    if (lhsState.dimIsStructured(i) && rhsState.dimIsStructured(i)) {
      auto newOffset =
          addOFRs(lhsOffsets[i], rhsOffsets[i], loc, builder);
      offsets.push_back(newOffset);
      auto newStride = addOFRs(lhsState.axes[i].stride, rhsState.axes[i].stride,
                               loc, builder);
      strides.push_back(newStride);
    } else if (mergeDimWithUnstructured(lhsState, rhsState, lhsOffsets, i,
                                        offsets, strides,
                                        isAnalysisingUnstructured, op, builder,
                                        refusal)
                   .failed()) {
      return failure();
    }

    sizes.push_back(lhsState.axes[i].size);
  }

  // A WRAP ON THE SUM SWALLOWS EVERY AXIS, so the only thing that can join it
  // is a term that lands OUTSIDE -- a scalar, which rides base_offset. Anything
  // with an axis of its own would have to go inside a modulo it was not written
  // under.
  if (lhsState.flatShape || rhsState.flatShape) {
    if (lhsState.flatShape && rhsState.flatShape) {
      return refuse(op, refusal,
                    "do not support adding two states that both wrap the sum");
    }
    const PtrState &flat = lhsState.flatShape ? lhsState : rhsState;
    const PtrState &other = lhsState.flatShape ? rhsState : lhsState;
    for (uint64_t i = 0; i < other.getRank(); i++) {
      if (!hasConstZero(other.axes[i].offset) ||
          !hasConstZero(other.axes[i].stride)) {
        return refuse(
            op, refusal,
            "do not support adding an axis term to a state that wraps the sum");
      }
    }
    flatShape = flat.flatShape;
  }

  // ONE WRAP PER AXIS, AND THE AXES ARE INDEPENDENT. `shape` has a slot per
  // axis and each says `(off_d + p_d*stride_d) mod W_d`, so two states wrapping
  // DIFFERENT axes merge axis by axis. Only two moduli on the SAME axis have no
  // answer -- and this used to refuse the pair outright, one `hasModulo()`
  // above a loop that was already per-axis.
  auto indexTy = IndexType::get(op->getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  for (uint64_t i = 0; i < lhsState.getRank(); i++) {
    if (!lhsState.dimIsStructured(i) || !rhsState.dimIsStructured(i)) {
      // Shape is always 0 for non-structured dimension.
      shape.push_back(index0);
      continue;
    }

    bool lhsWraps = lhsState.dimHasModulo(i);
    bool rhsWraps = rhsState.dimHasModulo(i);
    if (lhsWraps && rhsWraps) {
      return refuse(op, refusal, "do not support two moduli on one axis");
    }
    if (!lhsWraps && !rhsWraps) {
      shape.push_back(index0);
      continue;
    }

    PtrState const *wrapped = lhsWraps ? &lhsState : &rhsState;
    PtrState const *other = lhsWraps ? &rhsState : &lhsState;
    if (hasConstZero(other->axes[i].offset) &&
        hasConstZero(other->axes[i].stride)) {
      shape.push_back(wrapped->axes[i].shape);
    } else {
      return refuse(op, refusal,
                    "do not support adding to operand with modulo");
    }
  }


  assembleAxes(*this, lhsState, rhsState, offsets, sizes, strides, shape);
  return success();
}

void PtrState::dump() const {
  llvm::dbgs() << "PtrState: ";
  if (source) {
    llvm::dbgs() << "source: " << source << "\n";
  }
  if (scalar) {
    llvm::dbgs() << "scalar: " << scalar << "\n";
  }

  llvm::dbgs() << "offsets:\n";
  llvm::interleave(offsetsOf(), llvm::dbgs(), "\n");
  llvm::dbgs() << "\nstrides:\n";
  llvm::interleave(stridesOf(), llvm::dbgs(), "\n");
  llvm::dbgs() << "\nsizes:\n";
  llvm::interleave(sizesOf(), llvm::dbgs(), "\n");
  llvm::dbgs() << "\nshape:\n";
  llvm::interleave(shapeOf(), llvm::dbgs(), "\n");
  if (flatShape)
    llvm::dbgs() << "\nflat shape: " << flatShape;
  llvm::dbgs() << "\norder:\n";
  llvm::interleave(order, llvm::dbgs(), "\n");
  if (isStructured()) {
    llvm::dbgs() << "structured\n";
  } else {
    for (int i = 0; i < getRank(); i++) {
      llvm::dbgs() << "dim " << i;
      if (dimIsStructured(i))
        llvm::dbgs() << " structured\n";
      else
        llvm::dbgs() << " not strucuted\n";
    }
  }

  llvm::dbgs() << "\n";
}

LogicalResult PtrState::mulState(const PtrState &lhsIn, const PtrState &rhsIn,
                                 bool isAnalysisingUnstructured, Operation *op,
                                 OpBuilder &builder, Refusal *refusal) {
  assert(isEmpty());
  PtrState lhsState, rhsState;
  if (alignForMerge(lhsIn, rhsIn, lhsState, rhsState, op, builder, refusal)
          .failed())
    return failure();
  // BUILT COLUMN BY COLUMN, ASSEMBLED ONCE. The wrap is decided in a second
  // pass over the axes, so the four cannot be filled axis at a time.
  SmallVector<OpFoldResult> offsets, sizes, strides, shape;
  assert(lhsState.getRank() == rhsState.getRank());

  auto loc = op->getLoc();

  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  if (lhsState.source && rhsState.source) {
    return refuse(op, refusal, "do not support multiplying base pointers");
  }

  // currently do not support both tensors are effectively non-scalar
  if (!lhsState.scalar && !rhsState.scalar) {
    return refuse(op, refusal,
                  "only support multiplying pointer states when one of them "
                  "represent a scalar");
  }

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (!rhs->scalar && lhs->scalar) {
    std::swap(lhs, rhs);
  }

  if (lhsState.scalar && rhsState.scalar) {
    scalar =
        arith::MulIOp::create(builder, loc, lhsState.scalar, rhsState.scalar);
  }

  if (lhs->baseOffset)
    baseOffset = mulOFRs(lhs->baseOffset, rhs->scalar, loc, builder);
  // `(x mod W) * a == (x*a) mod (W*a)`, the identity the wrap slot already
  // uses.
  if (lhs->flatShape)
    flatShape = mulOFRs(lhs->flatShape, rhs->scalar, loc, builder);

  auto indexTy = IndexType::get(op->getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  for (uint64_t i = 0; i < lhs->axes.size(); i++) {
    if (lhs->dimIsStructured(i)) {
      OpFoldResult newOffset =
          mulOFRs(lhs->axes[i].offset, rhs->scalar, loc, builder);
      offsets.push_back(newOffset);
      OpFoldResult newStride =
          mulOFRs(lhs->axes[i].stride, rhs->scalar, loc, builder);
      strides.push_back(newStride);
      OpFoldResult newShape =
          mulOFRs(lhs->axes[i].shape, rhs->scalar, loc, builder);
      shape.push_back(newShape);
    } else {
      assert(!lhs->dimHasModulo(i) &&
             "should not have non-structured dimension with modulo");
      if (isAnalysisingUnstructured) {
        // Keep offsets as is for unstructured dimension.
        // The address calculation will be done later in structured to
        // memref pass.
        offsets.push_back(lhs->axes[i].offset);
        // Mul the scalar to stride.
        OpFoldResult newStride =
            mulOFRs(lhs->axes[i].stride, rhs->scalar, loc, builder);
        strides.push_back(newStride);
      } else {
        // Just propagate the unstructured offset to the result to track the
        // unstructured dimension. The real address calculation will be done
        // later in the PtrAnalysis::visitOperandAddptr.
        OpFoldResult newOffset = lhs->axes[i].offset;
        offsets.push_back(newOffset);
        // Mul the scalar to stride.
        OpFoldResult newStride = lhs->axes[i].stride;
        strides.push_back(newStride);
      }
      // Shape is always 0 for non-structured dimension.
      shape.push_back(index0);
    }
    sizes.push_back(lhs->axes[i].size);
  }

  // AN INVARIANT GUARD, NOT A CAPABILITY BOUNDARY -- and saying which is the
  // whole content of this. Everything above scales by `rhs->scalar`, which is
  // the value BEFORE any wrap of its own; a state that carries both is
  // inconsistent, and scaling by it drops the modulo silently.
  //
  //     measured   `r * (splat(a) % 5)` with this removed gives
  //                `strides: [%arg2]` -- the stride is `a`, not `a % 5`.
  //
  // THE INVARIANT IS THAT THE TWO NEVER MEET. `scalar` is set at five places
  // (a scalar operand, a splat, a const splat, addState and mulState when BOTH
  // sides have one, and reconcileLoopPtrState at rank 0), and every one of them
  // leaves a state that is rank 0 or has every stride at a constant 0. A wrap
  // needs an axis that CONTRIBUTES, and visitOperandRemAsWrap folds a modulo
  // that reaches no axis into the value instead of parking it -- which is what
  // closed the one route that did reach here.
  //
  //     measured   0 over the lit suite and the kernels of
  //                pytorchsim-triton-opt (312 and 430 on 2026-09-02), with a
  //                control: reverting that fold makes the probe fire above.
  //
  // So this is kept as the guard for a state nobody should build, not as a
  // shape to teach. If it fires, the bug is upstream of here.
  if (rhs->hasModulo()) {
    return refuse(op, refusal,
                  "multiplying by a state that carries a modulo -- its scalar "
                  "is the value before the wrap");
  }


  assembleAxes(*this, lhsState, rhsState, offsets, sizes, strides, shape);
  return success();
}

LogicalResult PtrState::mergeUnstructuredState(const PtrState &other,
                                                 Operation *op,
                                                 Refusal *refusal) {
  // A GATHER INDEX IS ONE VALUE PER ELEMENT OF THE GATHERED AXIS, and a split
  // axis has no single such count -- the values are data, so they cannot be
  // decomposed the way an affine walk can. A stepped axis meeting a gather is
  // therefore a demotion, not a split.
  for (auto n : llvm::concat<const int32_t>(split, other.split))
    if (n > 1) {
      return refuse(op, refusal, "a split axis cannot carry a gather index");
    }
  // A STEP IS THE SAME REFUSAL. `x % W` and `x // K` off one value reach the
  // address as two terms, and when the `//` half is structured the `%` half is
  // what gathers -- so the merge happens HERE and not at an add. Letting it
  // through drops the gathered term and the address is silently short one.
  if (hasStep() || other.hasStep()) {
    return refuse(op, refusal, "a stepped axis cannot carry a gather index");
  }
  if (isStructured() || other.isStructured()) {
    return refuse(op, refusal,
                  "Expect merging pointer states both of which are "
                  "unstructured, but got structured state");
  }
  if (other.getNonStructuredDim() != getNonStructuredDim()) {
    return refuse(op, refusal,
                  "do not support merging pointer states with different "
                  "non-structured dimensions");
  }
  if (getRank() != other.getRank()) {
    return refuse(op, refusal,
                  "do not support merging pointer states with different ranks");
  }
  int gatherDim = other.getNonStructuredDim();

  // Merge gatherDim data from other.
  axes[gatherDim].offset = other.axes[gatherDim].offset;
  axes[gatherDim].stride = other.axes[gatherDim].stride;
  axes[gatherDim].shape = other.axes[gatherDim].shape;

  return success();
}

// Every size of a state that is about to become an op is static: the analysis
// only ever builds one from a shaped type it has already read.
SmallVector<int64_t> PtrState::getStaticSizes() const {
  SmallVector<int64_t> staticSizes;
  for (size_t i = 0; i < getRank(); i++) {
    auto s = getIntAttr(axes[i].size);
    assert(s.has_value());
    staticSizes.push_back(s.value());
  }
  return staticSizes;
}

// THE BASE OFFSET RIDES THE OP'S OWN SLOT, IN BOTH CREATORS, UNCONDITIONALLY.
// It belongs to no axis -- it is what a splat adds to every element -- and by
// here there is nothing left to decide: 3500149 gave it a slot and 544a71f
// stopped it being folded onto an axis on the way.
//
// WHY IT MUST NOT BE FOLDED HERE. The only axes left to fold onto may wrap, and
// a term put inside a modulo that is a multiple of the wrap CANCELS -- a loop
// advancing a wrapped pointer then reads one address forever. The gather
// creator cannot fold either, for a different reason: the lowering overwrites
// the gather dim's offset with the index tensor.
//
// TWO PLACES UPSTREAM DO FOLD ONE, and neither contradicts this. visitOperandRem
// folds it onto the axis it is about to wrap, because a base written INSIDE the
// modulo expression belongs inside it; visitOperandExpandDims folds it onto dim
// 0 before inserting an axis, because after the insert "dim 0" names something
// else. Both fold a base that is already on that axis's side of the expression.
Value PtrState::getCarriedBaseOffset(OpBuilder &builder, Location loc) const {
  if (!baseOffset)
    return Value();
  return getValueOrCreateConstantIndexOp(builder, loc, baseOffset);
}

tts::MakeTensorPtrOp PtrState::createTTSMakeTensorPtrOp(OpBuilder &builder,
                                                        Location loc,
                                                        Type replaces) const {
  auto sizes = getStaticSizes();
  auto op = mlir::tts::MakeTensorPtrOp::create(
      builder, loc, source, sizes, stridesOf(), offsetsOf(), shapeOf(), order,
      getCarriedBaseOffset(builder, loc),
      flatShape ? ofrToIndexValue(flatShape, loc, builder) : Value(),
      triton::encodingReplacing(replaces, sizes));
  if (hasStep())
    op.setStaticStepAttr(builder.getDenseI64ArrayAttr(stepsOf()));
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_tensor_ptr:\n";
    op->dump();
  });

  return op;
}

// The index tensor is the gathered axis's offset, so it has to be that axis's
// SHAPE. Nothing else in the state changes -- the collapse is the whole reason
// this is a step of its own and not part of the creator below.
void PtrState::collapseGatherOffsetTo1D(OpBuilder &builder, Location loc) {
  int nonContinuousDim = getNonStructuredDim();
  Value nonContinuousOffset = cast<Value>(axes[nonContinuousDim].offset);

  auto offsetTy = cast<ShapedType>(nonContinuousOffset.getType());
  if (offsetTy.getRank() <= 1)
    return;

  SmallVector<ReassociationExprs, 4> reassociationMap(1);
  for (int i = 0; i < offsetTy.getRank(); ++i)
    reassociationMap[0].push_back(builder.getAffineDimExpr(i));

  int offsetSize = 1;
  for (int size : offsetTy.getShape())
    offsetSize *= size;

  // The index keeps the offset's layout: a slice of it when only unit axes go,
  // the reshape's inferred layout otherwise.
  Attribute enc;
  if (auto ranked = dyn_cast<RankedTensorType>(offsetTy)) {
    SmallVector<int64_t> units, wide;
    for (int64_t d = 0; d < ranked.getRank(); d++)
      (ranked.getDimSize(d) == 1 ? units : wide).push_back(d);
    if (wide.size() <= 1) {
      if (wide.empty())
        units.pop_back();
      enc = triton::sliceEncoding(ranked.getEncoding(), units);
    } else {
      enc = triton::reshapeEncoding(ranked.getShape(), ranked.getEncoding(), {offsetSize});
    }
  }
  auto collapseTy =
      RankedTensorType::get({offsetSize}, offsetTy.getElementType(), enc);
  axes[nonContinuousDim].offset =
      tensor::CollapseShapeOp::create(builder, loc, collapseTy,
                                      nonContinuousOffset, reassociationMap)
          .getResult();
}

tts::MakeGatherScatterTensorPtrOp
PtrState::createTTSMakeGatherScatterTensorPtrOp(OpBuilder &builder,
                                                Location loc) const {
  // The gather pointer has no flat_shape slot, and dropping one silently is
  // what this whole slot exists to stop. Callers check before they get here.
  assert(!flatShape && "a sum wrap cannot ride a gather pointer");
  int nonContinuousDim = getNonStructuredDim();

  Value nonContinuousOffset = cast<Value>(axes[nonContinuousDim].offset);
  assert(cast<ShapedType>(nonContinuousOffset.getType()).getRank() <= 1 &&
         "collapseGatherOffsetTo1D has to run before this");

  // Generate tts::make_gather_scatter_tensor_ptr.
  auto op = mlir::tts::MakeGatherScatterTensorPtrOp::create(
      builder, loc, source, nonContinuousOffset, nonContinuousDim,
      getStaticSizes(), stridesOf(), offsetsOf(), shapeOf(),
      getCarriedBaseOffset(builder, loc));
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_gather_scatter_tensor_ptr:\n";
    op->dump();
  });

  return op;
}

// A PARKED INDEX IS AN IR VALUE AND CARRIES ITS OWN BASE. Every term in a
// state's baseOffset was found while visiting the very value being parked, so
// leaving it in the slot adds it a second time -- once inside the index, once
// beside it.
static void dropBaseOffsetTheIndexAlreadyCarries(PtrState &state) {
  state.baseOffset = OpFoldResult();
}

// One side wraps, the other gathers. A wrap ON THE AXIS ABOUT TO BE GATHERED
// has to be parked: that axis's `%` can only live as arithmetic inside the
// index tensor. A wrap on any other axis has a slot of its own on the pointer
// -- leaving it alone is what lets `addState` carry it there.
static LogicalResult parkWrapOnTheGatheredAxis(PtrState &lhsState,
                                               PtrState &rhsState, Value lhs,
                                               Value rhs, bool enableGS) {
  PtrState *wrapped;
  Value operand;
  if (!lhsState.isStructured() && rhsState.hasModulo()) {
    wrapped = &rhsState;
    operand = rhs;
  } else if (lhsState.hasModulo() && !rhsState.isStructured()) {
    wrapped = &lhsState;
    operand = lhs;
  } else {
    return success();
  }
  int gatherDim = (wrapped == &rhsState ? lhsState : rhsState)
                      .getNonStructuredDim();
  if (!wrapped->dimHasModulo(gatherDim))
    return success();
  if (!enableGS)
    return failure();
  dropBaseOffsetTheIndexAlreadyCarries(*wrapped);
  return wrapped->rebuildAsGatherScatter(operand, gatherDim);
}

// `(x op K) * scale` under a chain of shape ops, for op in div/rem. The
// MULTIPLY SITS ABOVE THE RESHAPE in what Inductor emits -- `28 * idx_h[:,
// None]` -- so both are peeled and the chain is compared op for op later.
namespace {
struct ScaledDivRem {
  Value x;
  int64_t k = 0, scale = 1;
  bool isRem = false, isSigned = false;
  SmallVector<Operation *> shape;   // outermost first
};
}  // namespace

static bool isShapeOp(Operation *op) {
  return isa_and_nonnull<triton::ExpandDimsOp, triton::BroadcastOp,
                         triton::ReshapeOp>(op);
}

static Value peelShapeOps(Value v, SmallVector<Operation *> &out) {
  while (auto *def = v.getDefiningOp()) {
    if (!isShapeOp(def))
      break;
    out.push_back(def);
    v = def->getOperand(0);
  }
  return v;
}

static std::optional<ScaledDivRem> peelScaledDivRem(Value v) {
  ScaledDivRem got;
  v = peelShapeOps(v, got.shape);
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    for (unsigned i = 0; i < 2; i++) {
      if (auto c = getConstantIntValue(mul->getOperand(i))) {
        got.scale = *c;
        v = mul->getOperand(1 - i);
        break;
      }
    }
  }
  v = peelShapeOps(v, got.shape);
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (isa<arith::RemSIOp, arith::DivSIOp>(def))
    got.isSigned = true;
  else if (!isa<arith::RemUIOp, arith::DivUIOp>(def))
    return std::nullopt;
  got.isRem = isa<arith::RemSIOp, arith::RemUIOp>(def);
  auto k = getConstantIntValue(def->getOperand(1));
  if (!k || *k <= 0)
    return std::nullopt;
  got.k = *k;
  got.x = def->getOperand(0);
  return got;
}

// WHAT A PEELED SHAPE OP DOES, not which value it did it to: the two sides
// differ in their operands by construction and in nothing else. An OUTER SUM
// -- `(i % K)[:, None] + (i // K)[None, :] * K` -- passes every other test.
static bool sameShapeChain(ArrayRef<Operation *> a, ArrayRef<Operation *> b) {
  if (a.size() != b.size())
    return false;
  for (auto [x, y] : llvm::zip(a, b)) {
    if (x->getName() != y->getName() ||
        x->getResult(0).getType() != y->getResult(0).getType() ||
        x->getAttrDictionary() != y->getAttrDictionary())
      return false;
  }
  return true;
}

// `(x / K) * K + (x % K)` is `x`, so the sum is the index the pair decomposed.
// THE SIGNEDNESS TRAVELS: the identity holds within signed and within
// unsigned, not across -- remsi is negative where divui reads huge.
static Value reflattenDivRem(arith::AddIOp addOp, OpBuilder &builder) {
  auto a = peelScaledDivRem(addOp.getLhs());
  auto b = peelScaledDivRem(addOp.getRhs());
  if (!a || !b || a->isRem == b->isRem)
    return nullptr;
  const ScaledDivRem &rem = a->isRem ? *a : *b;
  const ScaledDivRem &div = a->isRem ? *b : *a;
  if (rem.x != div.x || rem.k != div.k || rem.isSigned != div.isSigned)
    return nullptr;
  if (div.scale != rem.k * rem.scale)
    return nullptr;
  if (!sameShapeChain(rem.shape, div.shape))
    return nullptr;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(addOp);
  Location loc = addOp.getLoc();
  Value flat = rem.x;
  if (rem.scale != 1) {
    Value c = arith::ConstantOp::create(
                  builder, loc,
                  builder.getIntegerAttr(flat.getType(), rem.scale))
                  .getResult();
    flat = arith::MulIOp::create(builder, loc, flat, c).getResult();
  }
  for (Operation *sh : llvm::reverse(rem.shape)) {
    Operation *clone = builder.clone(*sh);
    clone->setOperand(0, flat);
    flat = clone->getResult(0);
  }
  return flat;
}

LogicalResult PtrAnalysis::visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  if (Value flat = reflattenDivRem(addOp, builder)) {
    demotionCounts.reflattenedDivRem++;
    return visitOperand(flat, state, loc, builder);
  }

  PtrState lhsState;
  if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (parkWrapOnTheGatheredAxis(lhsState, rhsState, addOp.getLhs(),
                                addOp.getRhs(),
                                enableMakeGatherScatterTensorPtr)
          .failed())
    return failure();
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.addState(lhsState, rhsState, isAnalysisingUnstructured, addOp,
                        builder, &mergeRefusal);
}

LogicalResult PtrAnalysis::visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(mulOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(mulOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (parkWrapOnTheGatheredAxis(lhsState, rhsState, mulOp.getLhs(),
                                mulOp.getRhs(),
                                enableMakeGatherScatterTensorPtr)
          .failed())
    return failure();

  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.mulState(lhsState, rhsState, isAnalysisingUnstructured, mulOp,
                        builder, &mergeRefusal);
}

// Every term seen so far is inside the wrap -- `%` takes the whole expression
// built up to here -- so a base offset goes onto the axis that receives the
// modulus before the modulus is written.
// A `%` ON A STRUCTURED AXIS IS A WRAP WITH A SLOT OF ITS OWN, and both passes
// read it that way. The unstructured pass parks the whole `%` as one more index
// term only when that reading fails -- claiming a structured axis would leave
// TWO axes unstructured, which mergeUnstructuredState refuses.
// Through the shape ops AND the width casts to the value that STATES
// something: what this asks is an index identity, and a zext of an index is
// the same index.
static Value peelThru(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (isShapeOp(def) || isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(def))
      v = def->getOperand(0);
    else
      break;
  }
  return v;
}

// `{value: B}` for every `value < B` this mask asserts, `and` included. A BOUND
// ON A QUOTIENT IS ONE ON THE DIVIDEND -- `(x // K) < c` is `x < c*K` for
// positive c and K -- so the walk peels the divisions it is written over.
static void maskBounds(Value mask, llvm::DenseMap<Value, int64_t> &out,
                       unsigned depth = 0) {
  if (depth > 8)
    return;
  Operation *def = peelThru(mask).getDefiningOp();
  if (!def)
    return;
  if (isa<arith::AndIOp>(def)) {
    for (Value o : def->getOperands())
      maskBounds(o, out, depth + 1);
    return;
  }
  auto cmp = dyn_cast<arith::CmpIOp>(def);
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
    return;
  auto b = getConstantIntValue(cmp.getRhs());
  Value v = peelThru(cmp.getLhs());
  for (unsigned i = 0; b && i < 8; i++) {
    auto it = out.find(v);
    if (it == out.end() || it->second > *b)
      out[v] = *b;
    auto div = v.getDefiningOp<arith::DivSIOp>();
    if (!div || *b <= 0)
      break;
    auto k = getConstantIntValue(div.getRhs());
    if (!k || *k <= 0)
      break;
    b = *b * *k;
    v = peelThru(div.getLhs());
  }
}

// `a < B` gives `a // K < ceil(B / K)`, for as many divisions as chain.
static void carryThroughDiv(llvm::DenseMap<Value, int64_t> &bound,
                            Operation *scope) {
  for (unsigned round = 0; round < 4; round++) {
    scope->walk([&](Operation *op) {
      if (!isa<arith::DivSIOp, arith::DivUIOp>(op))
        return;
      auto k = getConstantIntValue(op->getOperand(1));
      auto src = bound.find(peelThru(op->getOperand(0)));
      if (!k || *k <= 0 || src == bound.end())
        return;
      int64_t got = (src->second + *k - 1) / *k;
      auto it = bound.find(op->getResult(0));
      if (it == bound.end() || it->second > got)
        bound[op->getResult(0)] = got;
    });
  }
}

// Every value `op` carries `value` into, or nullopt if it cannot be crossed.
// scf.for binds init operands[3+j], block args[1+j] and results[j] at one j;
// any other region is crossed only when it holds no access of its own.
static std::optional<SmallVector<Value>> forwardsInto(Operation *op,
                                                      Value value) {
  SmallVector<Value> out;
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    for (auto [j, init] : llvm::enumerate(forOp.getInitArgs()))
      if (init == value) {
        out.push_back(forOp.getRegionIterArgs()[j]);
        out.push_back(forOp.getResult(j));
      }
    return out;
  }
  for (Region &r : op->getRegions()) {
    bool hasAccess = false;
    r.walk([&](Operation *i) {
      if (isa<triton::LoadOp, triton::StoreOp>(i))
        hasAccess = true;
    });
    if (hasAccess)
      return std::nullopt;
  }
  for (Value r : op->getResults())
    out.push_back(r);
  return out;
}

// Every tt.load/tt.store that reads `value`, or nullopt if not knowable. NOT A
// SHORTER LIST: the caller asks whether EVERY reader bounds the modulo, and a
// missing one is the reader that answers no.
static std::optional<SmallVector<Operation *>> accessesOf(Value value) {
  llvm::SmallPtrSet<Value, 16> seen{value};
  SmallVector<Value> front{value};
  SmallVector<Operation *> out;
  for (unsigned depth = 0; !front.empty(); depth++) {
    if (depth > 40)
      return std::nullopt;
    SmallVector<Value> next;
    for (Value v : front)
      for (Operation *user : v.getUsers()) {
        if (isa<triton::LoadOp, triton::StoreOp>(user)) {
          out.push_back(user);
          continue;
        }
        auto fwd = forwardsInto(user, v);
        if (!fwd)
          return std::nullopt;
        for (Value r : *fwd)
          if (seen.insert(r).second)
            next.push_back(r);
      }
    front = std::move(next);
  }
  return out;
}

// `x % W` is `x` where the mask of EVERY access that reads it says `x < W`.
// A MASKED-OFF POSITION IS ONE THE ACCESS DOES NOT TOUCH, so its address never
// had to be in range; leaving the modulo standing ends the structured walk and
// the access becomes a gather over an index it did not need.
// `x % W` IS `x` WHERE EVERY ACCESS THAT READS IT IS MASKED TO `x < W`, so the
// remainder is dropped before the analysis rather than read through: leaving
// the op standing feeds it to everyone ELSE who reads it, and a modulo the
// structured walk cannot take makes the access a gather over an index it did
// not need. What a masked-off position addresses is the backend's to place.
static bool modIsIdentityUnderMasks(Operation *remOp);

void PtrAnalysis::dropMaskedModulo(Operation *scope) {
  SmallVector<Operation *> gone;
  scope->walk([&](Operation *op) {
    if (isa<arith::RemSIOp, arith::RemUIOp>(op) && modIsIdentityUnderMasks(op))
      gone.push_back(op);
  });
  for (Operation *op : gone) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    op->erase();
  }
}

static bool modIsIdentityUnderMasks(Operation *remOp) {
  const bool dbg = ::getenv("TS_MODDBG") != nullptr;
  auto say = [&](const char *why) {
    if (dbg)
      llvm::errs() << "[modid] no: " << why << "\n";
    return false;
  };
  auto w = getConstantIntValue(remOp->getOperand(1));
  if (!w || *w <= 0)
    return say("modulus not a positive constant");
  Value x = peelThru(remOp->getOperand(0));
  auto readers = accessesOf(remOp->getResult(0));
  if (!readers)
    return say("readers not knowable");
  if (readers->empty())
    return say("no reader");
  Operation *scope = remOp->getParentOfType<triton::FuncOp>();
  if (!scope)
    return say("no enclosing tt.func");
  for (Operation *access : *readers) {
    Value mask;
    if (auto load = dyn_cast<triton::LoadOp>(access))
      mask = load.getMask();
    else
      mask = cast<triton::StoreOp>(access).getMask();
    if (!mask)
      return say("a reader has no mask");
    llvm::DenseMap<Value, int64_t> bound;
    maskBounds(mask, bound);
    carryThroughDiv(bound, scope);
    auto it = bound.find(x);
    if (it == bound.end())
      return say("the mask bounds nothing about the dividend");
    if (it->second > *w) {
      if (dbg)
        llvm::errs() << "[modid] no: bound " << it->second << " > W " << *w
                     << "\n";
      return false;
    }
  }
  if (dbg)
    llvm::errs() << "[modid] YES, W=" << *w << "\n";
  return true;
}

LogicalResult PtrAnalysis::visitOperandRem(Operation *op, Value lhs, Value rhs,
                                           Value result, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());
  if (!isAnalysisingUnstructured)
    return visitOperandRemAsWrap(op, lhs, rhs, result, state, loc, builder);

  assert(enableMakeGatherScatterTensorPtr &&
         "PtrAnalysis: isAnalysisingUnstructured should only be true "
         "when enableMakeGatherScatterTensorPtr is true");
  PtrState wrap;
  if (visitOperandRemAsWrap(op, lhs, rhs, result, wrap, loc, builder)
          .succeeded()) {
    state = std::move(wrap);
    return success();
  }
  return recordDemotion(Demoted::Rem, state.rebuildAsUnsupportedOp(result),
                        result);
}

// EVERY ELEMENT THIS STATE DESCRIBES IS >= 0. Asked term by term: an axis
// walks [offset, offset + (size-1)*stride], so a non-negative offset and a
// non-negative stride keep the whole walk on that side.
static bool stateIsNonNegative(const PtrState &state) {
  if (!nonNegativeOFR(state.baseOffset))
    return false;
  return llvm::all_of(state.axes, [](const PtrState::Axis &ax) {
    return nonNegativeOFR(ax.offset) && nonNegativeOFR(ax.stride);
  });
}

LogicalResult PtrAnalysis::visitOperandRemAsWrap(Operation *op, Value lhs,
                                                 Value rhs, Value result,
                                                 PtrState &state,
                                                 const Location loc,
                                                 OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState rhsState;
  if (visitOperand(rhs, rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    return refuse(op,
                  "only support cases when rhs of remainder contains scalar");
  }

  if (visitOperand(lhs, state, loc, builder).failed()) {
    return failure();
  }

  // When lhs already not structured, just build state from current op.
  if (!state.isStructured()) {
    dropBaseOffsetTheIndexAlreadyCarries(state);
    return state.rebuildAsGatherScatter(result,
                                        state.getNonStructuredDim());
  }

  // If there are multiple modulo ops on an expression (e.g.: (a % b) % c), we
  // would have already populated the modulo states after visiting the lhs.
  // Assert that all the modulo states are empty.
  // ONE MODULO PER ADDRESS. `(a % b) % c` would put a wrap inside a wrap, and
  // neither slot says that: `shape` is one modulus per axis and `flat_shape`
  // one on their sum, both read as the OUTERMOST thing that happened. At rank 1
  // the index table carries the composition instead; above it there is nowhere
  // to put it.
  if (state.hasModulo()) {
    if (state.getRank() == 1 && enableMakeGatherScatterTensorPtr) {
      dropBaseOffsetTheIndexAlreadyCarries(state);
      return state.rebuildAsGatherScatter(result, 0);
    }
    return refuse(op, "a second modulo on an address that already wraps, and "
                      "no axis is free to carry it");
  }



  auto foldBaseIntoWrap = [&](uint64_t wrapped) {
    if (!state.baseOffset)
      return;
    state.axes[wrapped].offset =
        addOFRs(state.axes[wrapped].offset, state.baseOffset, loc, builder);
    state.baseOffset = OpFoldResult();
  };

  // THE WRAPPED AXIS IS THE ONE THAT REACHES THE SUM the `%` is taken of, at
  // any rank. An axis reaches it only by CONTRIBUTING: a size-1 axis holds one
  // element and a stride-0 axis does not move, so neither adds anything the
  // modulo can act on. The wrap goes on the single axis that does contribute,
  // and two of those do not say which. Read from the STATE, which has a size
  // and a stride per axis at every rank -- the op's result type is not a tensor
  // when the `%` is on a scalar.
  if (state.getRank() == 0) {
    return refuse(op, "a modulo on a scalar wraps no axis");
  }

  // AND A WALK THAT IS WHOLE LAPS DOES NOT REACH IT EITHER. `(x + n*K) mod K`
  // is `x mod K`, so an axis the modulus divides both ways is invisible to the
  // `%` and is left standing still. The sign is the soundness, the same one a
  // bit mask rests on: `remsi` truncates toward zero, so `(-3 + 8) rem 8` is 5
  // while `-3 rem 8` is -3. THE BASE OFFSET IS LEFT ALONE -- the identity holds
  // for it too and the corpus cannot tell the two answers apart, so the arm
  // would only retire a wrap a later cut still has to carry an offset through.
  if (auto k = getConstantIntValue(OpFoldResult(rhsState.scalar));
      k && *k > 1 && stateIsNonNegative(state)) {
    for (auto &ax : state.axes) {
      if (dividesOFR(ax.offset, *k) && dividesOFR(ax.stride, *k)) {
        ax.offset = builder.getIndexAttr(0);
        ax.stride = builder.getIndexAttr(0);
      }
    }
  }

  auto opShape = state.getStaticSizes();
  int wrapped = -1;
  bool twoWide = false;
  for (int i = 0; i < (int)opShape.size(); i++) {
    auto stride = getIntAttr(state.axes[i].stride);
    if (opShape[i] <= 1 || (stride && *stride == 0))
      continue;
    if (wrapped >= 0) {
      twoWide = true;
      break;
    }
    wrapped = i;
  }
  // NO AXIS REACHES IT AT ALL means every element holds the same value, so the
  // `%` is on a CONSTANT. Fold it into that value and leave no wrap behind: a
  // shape on an axis that does not move is a modulo no descriptor can walk, and
  // it leaves `scalar` holding the PRE-wrap number -- which is why mulState has
  // to refuse a wrapped operand rather than scale by it.
  if (wrapped < 0) {
    OpFoldResult total = state.baseOffset;
    for (auto &ax : state.axes)
      total = total ? addOFRs(total, ax.offset, loc, builder) : ax.offset;
    total = remOFRs(total, rhsState.scalar, loc, builder);
    for (auto &ax : state.axes)
      ax.offset = builder.getIndexAttr(0);
    state.baseOffset = total;
    if (state.scalar)
      state.scalar = ofrToIndexValue(total, loc, builder);
    return success();
  }

  // TWO WIDE AXES MEANS THE `%` IS ON THE SUM, not on either of them. It has a
  // slot of its own: every axis then goes inside it, which is why the base
  // folds onto an axis here exactly as it does for a per-axis wrap.
  foldBaseIntoWrap(wrapped);
  if (twoWide) {
    state.flatShape = rhsState.scalar;
    return success();
  }

  // AND A WRAP THAT CANNOT FIRE IS NOT A WRAP. This is the sibling of the
  // `wrapped < 0` arm above: there no axis reaches the `%`, here one does and
  // never crosses it. The lhs's range is final at this line -- every later term
  // is added OUTSIDE the modulo -- so consuming it now is what the slot means.
  if (triton::wrapNeverFires(OpFoldResult(rhsState.scalar),
                             state.axes[wrapped].offset,
                             state.axes[wrapped].stride, opShape[wrapped]))
    return success();

  state.axes[wrapped].shape = rhsState.scalar;
  return success();
}

// `x & (2^k - 1)` KEEPS THE LOW k BITS, which is `x % 2^k` while x cannot go
// negative. Writes the two operands of that remainder and answers whether the
// `and` is one; `k >= 1` because a modulo of 1 is the constant 0.
static bool bitMaskAsModulo(arith::AndIOp andOp, Value &dividend,
                            Value &modulus, const Location loc,
                            OpBuilder &builder) {
  // ASKED OF THE TYPE: `and` is one op at every width and means logical and at
  // i1, where the low bit is the value itself and no address is involved.
  auto ty = andOp.getType();
  auto elemTy = isa<ShapedType>(ty) ? cast<ShapedType>(ty).getElementType() : ty;
  if (!isa<IntegerType>(elemTy) || elemTy.getIntOrFloatBitWidth() <= 1)
    return false;
  // EITHER SIDE MAY CARRY THE MASK -- `and` is commutative and nothing on the
  // way here puts constants on the right.
  for (unsigned i : {1u, 0u}) {
    auto m = triton::constantIntOrSplat(andOp->getOperand(i));
    if (!m || *m <= 0 || ((*m + 1) & *m) != 0)
      continue;
    Value x = andOp->getOperand(1 - i);
    if (!isNonNegative(x))
      continue;
    dividend = x;
    modulus = arith::ConstantIndexOp::create(builder, loc, *m + 1);
    return true;
  }
  return false;
}

// AN UNSIGNED `//` OR `%` OF A NON-NEGATIVE VALUE IS THE SIGNED ONE. The two
// agree on every value whose sign bit is clear, and a bit pattern this analysis
// would read as negative is not an address either way. Routed to the signed
// arms so a `//` has one reading and a `%` has one, rather than two that drift.
LogicalResult PtrAnalysis::visitOperandUnsigned(Operation *op, PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {
  assert(state.isEmpty());
  bool isRem = isa<arith::RemUIOp>(op);
  Value lhs = op->getOperand(0), rhs = op->getOperand(1);
  Value result = op->getResult(0);
  if (isNonNegative(lhs) && isNonNegative(rhs))
    return isRem ? visitOperandRem(op, lhs, rhs, result, state, loc, builder)
                 : visitOperandDiv(op, lhs, rhs, result, state, loc, builder);
  if (!enableMakeGatherScatterTensorPtr)
    return failure();
  return recordDemotion(isRem ? Demoted::Rem : Demoted::Div,
                        state.rebuildAsUnsupportedOp(result), result);
}

// A BIT MASK IS A WRAP, and reading it as one is all this arm adds: what it
// cannot read leaves by the same exit an `and` took before the arm existed.
LogicalResult PtrAnalysis::visitOperandAnd(arith::AndIOp andOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());
  Value dividend, modulus;
  if (bitMaskAsModulo(andOp, dividend, modulus, loc, builder)) {
    if (!isAnalysisingUnstructured)
      return visitOperandRemAsWrap(andOp, dividend, modulus,
                                   andOp.getResult(), state, loc, builder);
    PtrState wrap;
    if (visitOperandRemAsWrap(andOp, dividend, modulus, andOp.getResult(),
                              wrap, loc, builder)
            .succeeded()) {
      state = std::move(wrap);
      return success();
    }
  }
  if (!enableMakeGatherScatterTensorPtr)
    return failure();
  return recordDemotion(Demoted::Chosen,
                        state.rebuildAsUnsupportedOp(andOp.getResult()),
                        andOp.getResult());
}

// The state describes the EXTENDED value, so a per-element tensor offset it
// carries must be widened here. Fixed where the width is lost, not where a
// later consumer trips on it.
LogicalResult PtrAnalysis::visitOperandExtSI(arith::ExtSIOp extOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());
  if (visitOperand(extOp.getIn(), state, loc, builder).failed())
    return failure();

  // THE STATE STANDS FOR THE WIDE VALUE, so it has to carry the wide tensor:
  // visiting the input alone leaves it holding the narrow one.
  if (auto resTy = dyn_cast<RankedTensorType>(extOp.getType()))
    state.widenOffsets(resTy.getElementType(), builder, loc);
  return success();
}

// DOES EVERY ELEMENT OF THIS STATE HOLD THE SAME VALUE? A splat does, and so
// does a scalar: no axis moves and none is folded or stepped.
static bool stateStandsStill(const PtrState &st) {
  return llvm::all_of(st.axes, [](const PtrState::Axis &ax) {
    return hasConstZero(ax.stride) && hasConstZero(ax.offset) &&
           hasConstZero(ax.shape) && ax.step == 0;
  });
}

// THE RANGE EVERY ELEMENT OF A TENSOR STATE FALLS IN. An axis walks
// [offset, offset + (size-1)*stride]; a wrap or a step makes the walk something
// else and a scalar is not this question, so all three answer "unknown".
static std::optional<std::pair<int64_t, int64_t>>
tensorValueRange(const PtrState &st) {
  if (st.scalar || st.getRank() == 0 || !st.isStructured())
    return std::nullopt;
  int64_t lo = 0, hi = 0;
  if (st.baseOffset) {
    auto b = getConstantIntValue(st.baseOffset);
    if (!b)
      return std::nullopt;
    lo = hi = *b;
  }
  for (auto &ax : st.axes) {
    if (!hasConstZero(ax.shape) || ax.step != 0)
      return std::nullopt;
    auto o = getConstantIntValue(ax.offset);
    auto sd = getConstantIntValue(ax.stride);
    auto n = getConstantIntValue(ax.size);
    if (!o || !sd || !n || *n < 1)
      return std::nullopt;
    int64_t last = *o + (*n - 1) * *sd;
    lo += std::min(*o, last);
    hi += std::max(*o, last);
  }
  return std::make_pair(lo, hi);
}

// A CLAMP IS NOT AFFINE. `max(t, 4)` is 4 on the first four lanes and t after,
// and no offset/stride pair says that -- so a `make_tptr` cannot carry one and
// picking an operand to keep is a WRONG ADDRESS, not a conservative one. There
// are three honest answers and this gives all three.
LogicalResult PtrAnalysis::visitOperandClamp(Operation *op, bool isMin,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());
  Value lhs = op->getOperand(0), rhs = op->getOperand(1);
  Value result = op->getResult(0);

  PtrState lhsState, rhsState;
  if (visitOperand(lhs, lhsState, loc, builder).failed() ||
      visitOperand(rhs, rhsState, loc, builder).failed())
    return failure();

  // ONE: TWO NUMBERS CLAMP TO A NUMBER. Neither side varies, so the result does
  // not either -- mint the comparison and let the address ride on it.
  if (lhsState.scalar && rhsState.scalar && stateStandsStill(lhsState) &&
      stateStandsStill(rhsState)) {
    Value a = ofrToIndexValue(lhsState.scalar, loc, builder);
    Value b = ofrToIndexValue(rhsState.scalar, loc, builder);
    Value clamped = isMin ? arith::MinSIOp::create(builder, loc, a, b).getResult()
                          : arith::MaxSIOp::create(builder, loc, a, b).getResult();
    state = lhsState.getRank() >= rhsState.getRank() ? lhsState : rhsState;
    state.scalar = clamped;
    state.baseOffset = clamped;
    return success();
  }

  // TWO: A CLAMP THAT NEVER FIRES IS THE IDENTITY. `max(idx, 0)` where idx
  // cannot go below 0, `min(idx, N-1)` where its walk ends before N -- the
  // shape every bounds-safe kernel writes, and the only one that stays affine.
  //
  // AND THE TWO SIDES NEED DIFFERENT PROOFS ABOUT THE BOUND. `min(v, c)` is v
  // where v never reaches c, so a LOWER bound on c settles it and a runtime
  // `pid + 4096` has one. `max(v, c)` is v where v never falls to c, which
  // needs an UPPER bound, and a program id has none.
  auto clampIsIdentity = [&](const PtrState &walk, const PtrState &bnd) {
    if (!bnd.scalar || !stateStandsStill(bnd))
      return false;
    auto range = tensorValueRange(walk);
    if (!range)
      return false;
    if (isMin) {
      auto lo = lowerBoundOf(bnd.scalar);
      return lo.has_value() && range->second <= *lo;
    }
    auto hi = getConstantIntValue(bnd.scalar);
    return hi.has_value() && range->first >= *hi;
  };
  if (clampIsIdentity(lhsState, rhsState)) {
    state = lhsState;
    return success();
  }
  if (clampIsIdentity(rhsState, lhsState)) {
    state = rhsState;
    return success();
  }

  // THREE: AND OTHERWISE IT IS ONE ADDRESS PER ELEMENT. The clamp is real, so
  // the index table carries it -- which is also what makes the offsets of a
  // gather that clamps into range actually be in range.
  if (!enableMakeGatherScatterTensorPtr)
    return failure();
  return recordDemotion(Demoted::Chosen, state.rebuildAsUnsupportedOp(result),
                        result);
}


LogicalResult PtrAnalysis::visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                                 PtrState &state, Location loc,
                                                 OpBuilder &builder) {
  assert(state.isEmpty());

  auto tensorShape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + tensorShape[0] - 1) / tensorShape[0];
  assert(stride == 1 &&
         "Expect make_range op to always return tensor of stride 1");

  state.axes.push_back({builder.getIndexAttr(start),
                        builder.getIndexAttr(tensorShape[0]),
                        builder.getIndexAttr(stride), builder.getIndexAttr(0)});
  return success();
}

// expand_dims moves what the axes carry, so a base offset settles onto axis 0
// first and is shifted with it.
LogicalResult
PtrAnalysis::visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder) {
  assert(state.isEmpty());

  if (visitOperand(expandDimsOp.getSrc(), state, loc, builder).failed()) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  // A TENSOR OFFSET STANDS AT THE STATE'S RANK, SO IT MOVES WITH IT. An
  // unstructured dim's offset is one value per element of the tile, and
  // rebuildAsUnsupportedOp records it at whatever rank the value HAD when the
  // state fell unstructured. Leaving it behind gives one dim two spellings, and
  // addState then has to add `tensor<64xi32>` to `tensor<64x1xi32>` -- which is
  // resnet18's first conv. `tensor.expand_shape` and not `tt.expand_dims`
  // because the gather slot collapses the row back and MLIR folds that pair.
  for (auto &ax : state.axes)
    if (auto v = dyn_cast<Value>(ax.offset))
      if (auto ty = dyn_cast<RankedTensorType>(v.getType())) {
        if (ty.getRank() == 0)
          continue;
        SmallVector<int64_t> newShape(ty.getShape());
        newShape.insert(newShape.begin() + axis, 1);
        // Source dim `src` becomes result dim `src + (src >= axis)`, and the
        // unit axis joins the group next to it.
        SmallVector<ReassociationIndices> reassoc;
        for (int64_t src = 0; src < ty.getRank(); src++)
          reassoc.push_back({src + (src >= (int64_t)axis ? 1 : 0)});
        if (axis == 0)
          reassoc.front().insert(reassoc.front().begin(), 0);
        else
          reassoc[std::min<int64_t>((int64_t)axis - 1, ty.getRank() - 1)]
              .push_back(axis);
        Attribute enc;
        if (auto ranked = dyn_cast<RankedTensorType>(ty))
          enc = triton::reshapeEncoding(ranked.getShape(), ranked.getEncoding(), newShape);
        ax.offset = tensor::ExpandShapeOp::create(
                  builder, loc,
                  RankedTensorType::get(newShape, ty.getElementType(), enc), v,
                  reassoc)
                  .getResult();
      }

  // insert dimension info -- AT THE TENSOR AXIS'S STATE POSITION, which is the
  // axis number itself until some earlier axis has been split.
  auto at = state.axisRange(axis).first;
  expandTensorAxis(state, axis, at,
                   {builder.getIndexAttr(0), builder.getIndexAttr(1),
                    builder.getIndexAttr(0), builder.getIndexAttr(0)});

  return success();
}

LogicalResult
PtrAnalysis::visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                   PtrState &state, const Location loc,
                                   OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();

  if (!isa<ShapedType>(src.getType())) {
    return refuse(broadcastOp, "Unsupported broadcast source type");
  }

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  // A WIDENED AXIS HOLDS ONE VALUE REPEATED, so its stride is 0 and this says
  // so instead of inheriting whatever the size-1 axis carried. Today the two
  // agree -- the only size-1 axis with a non-zero stride is `make_range(0, 1)`,
  // which the pass folds to a splat before this is reached -- so writing the 0
  // changes no output and stops resting on that fold.
  //
  // AND THE AXIS IS THE TENSOR'S, NOT THE STATE'S. A split axis holds several
  // state axes, so `i` addresses a RANGE; a widened one is stride 0 throughout,
  // which makes its split meaningless, so the group collapses back to one axis.
  for (size_t i = 0; i < dstShape.size(); i++) {
    if (srcShape[i] == dstShape[i]) {
      continue;
    } else if (srcShape[i] < dstShape[i]) {
      auto [b, e] = state.axisRange(i);
      state.axes[b].size = builder.getIndexAttr(dstShape[i]);
      state.axes[b].stride = builder.getIndexAttr(0);
      state.axes[b].step = 0;   // stride 0 -- `(p//K)*0` is 0 whatever K is
      if (e > b + 1)
        collapseTensorAxisGroup(state, i, b, e);
    } else {
      llvm_unreachable("unexpected dimensions used in broadcast");
    }
  }
  return success();
}

// A splatted scalar belongs to no axis, so it goes to baseOffset rather than
// being parked on the outermost dim.
LogicalResult PtrAnalysis::visitOperandSplat(triton::SplatOp splatOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if (isa<IntegerType, IndexType, triton::PointerType>(src.getType())) {
    for (auto s : dstShape) {
      state.axes.push_back({builder.getIndexAttr(0), builder.getIndexAttr(s),
                            builder.getIndexAttr(0), builder.getIndexAttr(0)});
    }
  } else {
    return refuse(splatOp, "unsupported splat pattern");
  }

  if (state.scalar)
    state.baseOffset = state.scalar;

  return success();
}

// `p // K` WHERE K DOES NOT DIVIDE THE STRIDE. The axis is really two --
// p = outer*K + inner, and the value is `outer` -- so the state either GROWS an
// axis (a split, exact when K divides the extent) or records the PERIOD and
// lets the consumer walk the runs (a step, for when it does not). Everything
// here writes into `state`, and every exit that is not success leaves the
// division to `giveUp`, which demotes rather than fails.
LogicalResult PtrAnalysis::divideByReshapingTheAxis(
    PtrState &state, const PtrState &lhsState, int axis, int64_t k, int64_t s,
    int64_t n, Operation *divOp, OpBuilder &builder,
    llvm::function_ref<LogicalResult(const char *)> giveUp) {
  // `axis` is a STATE axis, and it is a tensor axis only while nothing has
  // split yet -- so a second split of one state is v1's refusal, not a
  // guess about which axis the number names.
  //
  // `x // K` DOES NOT RESTRUCTURE THIS ADDRESS: either the flag is off, or
  // this is the retry that follows a merge the restructuring made impossible.
  // THE TWO ARE ONE ANALYSIS. Leaving the split on for the retry left a split
  // axis facing the index table the un-minted step had become, and a merge has
  // no way to join those -- the address then fell TWO rungs, which is the very
  // thing the retry exists to prevent.
  if (!splitSteppedAxes || unmintDivision)
    return giveUp(unmintDivision
                      ? "x // K is un-minted here, because a merge could not "
                        "take what it built"
                      : "splitting is off");
  // AND THE ACCESS'S MASK HAS TO SURVIVE THE CUT. `axis` is a tensor axis
  // here -- nothing has split yet, the check below says so -- so it indexes
  // the box directly.
  if (splitCrossesLoop)
    return giveUp("this pointer crosses a loop boundary, where "
                  "get_structured_state counts its results by the tensor");
  if (axis < (int)unsplittableAxes.size() && unsplittableAxes[axis])
    return giveUp("the access's mask has a bound this axis cannot keep "
                  "across a cut");
  if (s != 1)
    return giveUp("the axis stride is not 1, so the index is not the divided value");
  if (!lhsState.split.empty())
    return giveUp("this state has already split an axis");
  state = lhsState;
  for (auto &ax : state.axes)
    ax.offset = builder.getIndexAttr(*getConstantIntValue(ax.offset) / k);
  if (state.baseOffset)
    state.baseOffset =
        divideExactly(state.baseOffset, k, divOp->getLoc(), builder);
  // outer counts the runs and is the quotient itself; inner stands still.
  // A wrap rides the OUTER half at W/k -- splitAxisBy refuses a wrapped
  // axis on purpose (alignSplits must not move one), so it is lifted off
  // here and put back where it belongs.
  auto w = getConstantIntValue(state.axes[axis].shape);
  state.axes[axis].shape = builder.getIndexAttr(0);
  if (splitAxisBy(state, axis, k, /*outer=*/1, /*inner=*/0, builder)
          .succeeded()) {
    if (w && *w)
      state.axes[state.axisRange(axis).first].shape =
          builder.getIndexAttr(*w / k);
    return success();
  }
  state.axes[axis].shape = w ? builder.getIndexAttr(*w) : state.axes[axis].shape;

  // BOTH OF THESE ARE PROPERTIES OF A STEPPED AXIS, so they gate the
  // composition below as much as a fresh step.
  if (w && *w)
    return giveUp("a stepped axis cannot also wrap");
  if (state.flatShape)
    return giveUp("a stepped axis cannot sit under a sum wrap");
  // (p // K1) // K2 == p // (K1*K2) -- compose rather than refuse, and
  // BEFORE the extent test: splitAxisBy refuses a stepped axis by rule, so
  // an extent that divides says nothing about THIS division.
  if (state.stepOf(axis)) {
    state.axes[axis].step *= k;
    return success();
  }
  // K DOES NOT DIVIDE THE EXTENT, so the runs are not all K long and no
  // uniform pair of axes says it. The step slot says the PERIOD instead
  // and p06 derives the runs, the way it already does from a wrap.
  if (n % k == 0)
    return giveUp("the extent divides but the split still refused");
  // A FRESH STEP. The period is all the slot carries; the consumer walks the
  // runs off it, 11, 11, 10 for 32 // 11.
  state.axes[axis].step = k;
  return success();
}

LogicalResult PtrAnalysis::visitOperandDiv(Operation *divOp, Value lhs,
                                           Value rhs, Value result,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(lhs, lhsState, loc, builder).failed())
    return failure();

  PtrState rhsState;
  if (visitOperand(rhs, rhsState, loc, builder).failed())
    return failure();

  // THE WHOLE EXPRESSION IS THE CONDITION. `(a + b) / K` is `a/K + b/K` only
  // when every term divides exactly, so one axis may vary and every constant
  // must be a multiple of K -- a guard on one axis would be unsound.
  //
  // AND A REFUSAL HERE IS A DEMOTION, NOT A FAILURE. Before this arm existed a
  // division took the no-arm exit, which records the value as an unstructured
  // offset; returning failure() instead would refuse the whole access.
  // THE REASON IS THE ONLY ACTIONABLE PART. "106 divisions demoted" says how
  // big the class is; "36 of them because the stride is not 1" says what to
  // build next, and each name below is one such thing.
  auto giveUp = [&](const char *why) -> LogicalResult {
    demotionCounts.divReasons[why]++;
    lastRefusal = why;
    if (!enableMakeGatherScatterTensorPtr)
      return failure();
    state = PtrState();
    return recordDemotion(Demoted::Div,
                          state.rebuildAsUnsupportedOp(result), result);
  };

  auto k = rhsState.scalar
               ? getConstantIntValue(OpFoldResult(rhsState.scalar))
               : std::nullopt;
  if (!k || *k <= 1 || lhsState.scalar || lhsState.source ||
      lhsState.flatShape)
    return giveUp("the divisor is not a constant > 1, or the value is a scalar, a base or sum-wrapped");
  // WHICH axis carries it is the actionable half: the split reshapes exactly
  // one axis, and every other axis in this arm has stride 0.
  for (auto [d, ax] : llvm::enumerate(lhsState.axes))
    if (!hasConstZero(ax.shape)) {
      if (hasConstZero(lhsState.axes[d].stride))
        return giveUp("a STILL axis carries a wrap");
      // A WRAP THE DIVISOR DIVIDES SURVIVES THE DIVISION, at W/k:
      //     (x mod W) / k  ==  (x / k) mod (W / k)      when k | W
      // Both arms below carry it, so only an indivisible one refuses here.
      auto w = getConstantIntValue(ax.shape);
      if (!w || *w % *k != 0)
        return giveUp("the WALKING axis carries a wrap the divisor does not "
                      "divide");
    }

  auto exact = [&](OpFoldResult ofr) {
    auto c = getConstantIntValue(ofr);
    return c && *c % *k == 0;
  };
  // WHETHER IT IS UNKNOWN OR MERELY UNALIGNED IS THE ACTIONABLE HALF: an
  // unaligned constant shifts the first run, a runtime one means nobody here
  // knows where the runs fall at all.
  if (lhsState.baseOffset && !dividesOFR(lhsState.baseOffset, *k))
    return giveUp(getConstantIntValue(lhsState.baseOffset)
                      ? "the base offset is a constant the divisor does not "
                        "divide"
                      : "the base offset is not known here");
  for (auto [o, _sz, _st, _sh, _k] : lhsState.axes)
    if (!exact(o))
      return giveUp("an axis offset is not a multiple of the divisor");

  int axis = -1;
  for (uint64_t i = 0; i < lhsState.axes.size(); i++) {
    if (hasConstZero(lhsState.axes[i].stride))
      continue;
    if (axis >= 0)
      return giveUp("more than one axis varies");
    axis = i;
  }

  // DECIDED BEFORE ANYTHING IS WRITTEN, so a refusal leaves the state clean for
  // the demotion above.
  std::optional<int64_t> s, n;
  if (axis >= 0) {
    s = getConstantIntValue(lhsState.axes[axis].stride);
    n = getConstantIntValue(lhsState.axes[axis].size);
    if (!s || !n)
      return giveUp("the axis extent or stride is not a constant");
    // THE OTHER CASE IS AN AXIS SPLIT. `p // K` is constant on runs of K, so
    // the axis is really two: p = outer*K + inner, and the value is `outer`.
    // The state then carries more axes than the tensor does, which is what
    // `split` records and what alignSplits reconciles at every add.
    if (*s % *k != 0)
      return divideByReshapingTheAxis(state, lhsState, axis, *k, *s, *n, divOp,
                                      builder, giveUp);
  }

  state = lhsState;
  auto scale = [&](OpFoldResult ofr) {
    return builder.getIndexAttr(*getConstantIntValue(ofr) / *k);
  };
  for (auto &ax : state.axes)
    ax.offset = scale(ax.offset);
  if (state.baseOffset)
    state.baseOffset =
        divideExactly(state.baseOffset, *k, divOp->getLoc(), builder);

  if (axis >= 0) {
    // THE AXIS ALREADY STEPS IN WHOLE BLOCKS, so the division is exact on it
    // too and no split is needed -- the shape a reshape would have written,
    // arriving as arithmetic instead.
    state.axes[axis].stride = builder.getIndexAttr(*s / *k);
    if (auto w = getConstantIntValue(state.axes[axis].shape))
      if (*w)
        state.axes[axis].shape = builder.getIndexAttr(*w / *k);
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandSub(arith::SubIOp subOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(subOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(subOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  // ONLY A SCALAR MAY BE SUBTRACTED, and a splat constant is the only shape
  // that reaches here: 32 of 32 subtractions in an address over tests/cases
  // have a constant rhs. NOT BECAUSE A NEGATIVE STRIDE IS UNSPELLABLE -- it
  // lowers (negative_stride_lowers.mlir) -- but because negating a state that
  // varies means negating every axis's stride and offset, and deciding what a
  // negated wrap is. Nothing asks.
  if (!rhsState.scalar || rhsState.source) {
    LLVM_DEBUG(subOp->emitRemark(
        "PtrAnalysis: only support subtracting a scalar from a pointer state"));
    // A REFUSAL IS A DEMOTION, the no-arm exit this used to take.
    if (!enableMakeGatherScatterTensorPtr)
      return failure();
    state = PtrState();
    return recordDemotion(Demoted::Sub,
                          state.rebuildAsUnsupportedOp(subOp.getResult()),
                          subOp.getResult());
  }

  OpFoldResult zero = builder.getIndexAttr(0);
  rhsState.scalar = getValueOrCreateConstantIndexOp(
      builder, loc, subOFRs(zero, OpFoldResult(rhsState.scalar), loc, builder));
  if (rhsState.baseOffset)
    rhsState.baseOffset = subOFRs(zero, rhsState.baseOffset, loc, builder);

  return state.addState(lhsState, rhsState, isAnalysisingUnstructured, subOp,
                        builder, &mergeRefusal);
}

LogicalResult PtrAnalysis::visitAddptrOnce(triton::AddPtrOp addptrOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState ptrState;
  if (visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), builder)
          .failed()) {
    return failure();
  } else if (!ptrState.source) {
    LLVM_DEBUG(llvm::dbgs()
               << "No src ptr state when processing " << addptrOp << "\n");
  }

  PtrState offsetState;
  if (visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(),
                   builder)
          .failed()) {
    return failure();
  }

  assert(ptrState.source && "ptr field should provide source / base pointer");

  // THE TENSOR RANKS ARE WHAT MUST AGREE; the state ranks part company as soon
  // as one side has split an axis, and addState below aligns them.
  assert(ptrState.tensorRank() == offsetState.tensorRank() &&
         "ptr and offset field should have the same rank");

  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }

  // A WRAP MEETING A GATHER HAS TO BE PARKED HERE TOO. visitOperandAdd and
  // visitOperandMul both do this and this one did not, so `p % W` added to an
  // already-gathered pointer by a `tt.addptr` -- which is how a two-term
  // address is usually spelled -- lost the modulo instead of moving it into
  // the index. The address then came out one term short and the answer was
  // wrong, not refused.
  if (parkWrapOnTheGatheredAxis(ptrState, offsetState, addptrOp.getPtr(),
                                addptrOp.getOffset(),
                                enableMakeGatherScatterTensorPtr)
          .failed())
    return failure();

  return state.addState(ptrState, offsetState, isAnalysisingUnstructured,
                        addptrOp, builder, &mergeRefusal);
}

LogicalResult PtrAnalysis::visitOperandAddptr(triton::AddPtrOp addptrOp,
                                              PtrState &state,
                                              const Location loc,
                                              OpBuilder &builder) {
  mergeRefusal = {};
  if (visitAddptrOnce(addptrOp, state, loc, builder).succeeded())
    return success();
  if (mergeRefusal.why)
    lastRefusal = mergeRefusal.why;

  // A STEP THAT CANNOT MERGE IS UN-MINTED, NOT DROPPED. Returning the failure
  // stops this address being rewritten at all and the gather-scatter arm in
  // rewriteAddptrOp never sees it, so the shape falls two rungs instead of one.
  // Visiting again with BOTH of `x // K`'s slots off is exactly the flag-off
  // analysis, which reaches an index table for these -- suppressing only the
  // step left the split standing and the merge still had nothing to join.
  if (unmintDivision || !mergeRefusal.fromStep)
    return failure();
  unmintDivision = true;
  state = PtrState();
  auto retry = visitAddptrOnce(addptrOp, state, loc, builder);
  unmintDivision = false;
  return retry;
}

LogicalResult PtrAnalysis::visitOperandConstSplat(arith::ConstantOp op,
                                                  PtrState &state,
                                                  const Location loc,
                                                  OpBuilder &builder) {
  assert(state.isEmpty());
  // this condition is to handle cases where tt.broadcast and tt.splat are
  // folded
  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  assert(attr.isSplat() && isa<IntegerType>(elementType));
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = builder.getIndexAttr(value.getSExtValue());
  auto constOp = arith::ConstantOp::materialize(builder, constAttr,
                                                builder.getIndexType(), loc);

  state.scalar = constOp;
  state.baseOffset = constOp.getResult();

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (size_t i = 0; i < resultType.getShape().size(); i++) {
    state.axes.push_back({builder.getIndexAttr(0),
                          builder.getIndexAttr(resultType.getShape()[i]),
                          builder.getIndexAttr(0), builder.getIndexAttr(0)});
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandMakeTPtr(tts::MakeTensorPtrOp makeTPtrOp,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {

  assert(state.isEmpty());
  state.source = makeTPtrOp.getBase();
  state.setAxes(makeTPtrOp.getMixedOffsets(), makeTPtrOp.getMixedSizes(),
                makeTPtrOp.getMixedStrides(), makeTPtrOp.getMixedShape());
  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());
  // AND THE TWO SLOTS THAT ARE NOT IN THE ARRAYS. Reading the op back has to
  // recover everything createTTSMakeTensorPtrOp put in it, or the state is a
  // DIFFERENT address than the op it was read from.
  if (Value carried = makeTPtrOp.getBaseOffset())
    state.baseOffset = carried;
  if (Value flat = makeTPtrOp.getFlatShape())
    state.flatShape = flat;

  return success();
}

LogicalResult PtrAnalysis::visitOperandForOp(scf::ForOp forOp, Value operand,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {

  auto it = llvm::find(forOp->getResults(), operand);
  auto index = std::distance(forOp->getResults().begin(), it);

  auto newState = getLoopResultPtrState(forOp, index);
  if (failed(newState)) {
    LLVM_DEBUG(forOp.emitWarning(
        "Rewrite for-op failed. Could not find PtrState returned by "
        "the loop."));
    return failure();
  }

  state = newState.value();
  return success();
}

LogicalResult PtrAnalysis::visitOperandIntToPtr(triton::IntToPtrOp op,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {
  state.source = op.getResult();
  return success();
}

LogicalResult PtrAnalysis::visitOperandBitcast(triton::BitcastOp op,
                                               PtrState &state,
                                               const Location loc,
                                               OpBuilder &builder) {
  auto resType = op.getResult().getType();
  if (isa<ShapedType>(resType)) {
    if (failed(visitOperand(op.getSrc(), state, loc, builder))) {
      return failure();
    }
    // A POINTER BITCAST MOVES NO ADDRESS, but it renames the UNIT every offset
    // is counted in, so crossing it multiplies the whole address expression by
    // the ratio -- and the SOURCE, whose pointee the descriptor will name.
    unsigned ratio = triton::ptrCastElementRatio(op);
    if (ratio == 0) {
      return refuse(
          op,
          "a pointer bitcast that widens an element would divide every offset");
    }
    if (ratio > 1) {
      // A GATHERED DIM IS NOT RESCALABLE HERE: its contribution is
      // offset * stride, and that offset is a per-element tensor that no
      // scalar factor multiplies (bitcast_keeps_the_chain.mlir @gathered_dim).
      for (uint64_t i = 0; i < state.axes.size(); i++) {
        if (!state.dimIsStructured(i)) {
          return refuse(
              op,
              "a pointer bitcast that rescales cannot cross a gathered dim");
        }
      }
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(op);
      OpFoldResult r = builder.getIndexAttr(ratio);
      // `(x mod W) * a == (x*a) mod (W*a)`, the identity mulState already uses.
      if (state.baseOffset)
        state.baseOffset = mulOFRs(state.baseOffset, r, loc, builder);
      if (state.flatShape)
        state.flatShape = mulOFRs(state.flatShape, r, loc, builder);
      for (auto &a : state.axes)
        for (auto *e : {&a.offset, &a.stride, &a.shape})
          *e = mulOFRs(*e, r, loc, builder);
    }
    if (state.source) {
      auto dstPointee =
          cast<triton::PointerType>(cast<ShapedType>(resType).getElementType())
              .getPointeeType();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(op);
      state.source =
          triton::BitcastOp::create(
              builder, loc, triton::PointerType::get(dstPointee, 1),
              state.source)
              .getResult();
    }
    return success();
  }
  state.source = op.getResult();
  return success();
}

LogicalResult PtrAnalysis::visitOperand(Value operand, PtrState &state,
                                        const Location loc,
                                        OpBuilder &builder) {
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  // Not using knownPtrs when isAnalysisingUnstructured is true.
  // This is because we are analyzing unstructured state, the data in knownPtrs
  // is not valid for unstructured state.
  if (!isAnalysisingUnstructured &&
      knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return success();
  }

  if (isa<IntegerType>(operand.getType())) {
    OpBuilder::InsertionGuard guard(builder);
    if (!isa<BlockArgument>(operand) && operand.getDefiningOp()) {
      builder.setInsertionPointAfter(operand.getDefiningOp());
    }
    auto castOp = arith::IndexCastOp::create(builder, loc,
                                             builder.getIndexType(), operand);
    state.scalar = castOp.getResult();
    return success();
  } else if (isa<IndexType>(operand.getType())) {
    state.scalar = operand;
    return success();
  }

  if (isa<triton::PointerType>(operand.getType())) {
    // A scalar pointer can either be produced by AddPtrOp or a block
    // argument
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        return visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc,
                                  builder);
      } else if (auto castOp = dyn_cast<triton::BitcastOp>(op)) {
        return visitOperandBitcast(castOp, state, loc, builder);
      } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
        return visitOperandIntToPtr(intToPtrOp, state, loc, builder);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        state.source = operand;
        return success();
      } else {
        return refuse(op, "Unexpected defining op for triton pointer operand");
      }
    } else {
      state.source = operand;
      return success();
    }
  }

  // A layout conversion changes where a tensor's elements sit, never what they are:
  // for an address it is the identity, so the analysis looks through it.
  if (auto cvt = operand.getDefiningOp<triton::gpu::ConvertLayoutOp>()) {
    return visitOperand(cvt.getSrc(), state, loc, builder);
  }
  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return visitOperandAdd(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<tts::MakeTensorPtrOp>()) {
    return visitOperandMakeTPtr(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::DivSIOp>()) {
    return visitOperandDiv(op, op.getLhs(), op.getRhs(), op.getResult(), state,
                           loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::SubIOp>()) {
    return visitOperandSub(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    return visitOperandMul(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return visitOperandMakeRange(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return visitOperandBroadcast(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return visitOperandSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return visitOperandExpandDims(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BitcastOp>()) {
    return visitOperandBitcast(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return visitOperandAddptr(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return visitOperandConstSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return visitOperandRem(op, op.getLhs(), op.getRhs(), op.getResult(), state,
                           loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AndIOp>()) {
    return visitOperandAnd(op, state, loc, builder);
  } else if (isa_and_nonnull<arith::RemUIOp, arith::DivUIOp>(
                 operand.getDefiningOp())) {
    return visitOperandUnsigned(operand.getDefiningOp(), state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return visitOperandExtSI(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MinSIOp>()) {
    return visitOperandClamp(op, /*isMin=*/true, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MaxSIOp>()) {
    return visitOperandClamp(op, /*isMin=*/false, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<scf::ForOp>()) {
    return visitOperandForOp(op, operand, state, loc, builder);
  } else if (!operand.getDefiningOp()) {
    if (!knownPtrs.contains(operand)) {
      return failure();
    }

    // This operand must be an iter-arg of an inner-loop in a multiple-level
    // nested loop, which means its PtrState must have already been populated
    // during rewriteForOp of the parent loop.
    state = knownPtrs[operand];
    return success();
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "PtrAnalysis: encountered addptr operand produced by an "
                  "unsupported operation: "
               << operand);

    if (!enableMakeGatherScatterTensorPtr) {
      LLVM_DEBUG(llvm::dbgs()
                 << "PtrAnalysis: failed to rebuild as unsupported op\n");
      return failure();
    }
    return recordDemotion(Demoted::Other,
                          state.rebuildAsUnsupportedOp(operand), operand);
  }
}

// THE MASK IS READ WHERE THE ADDRESS IS DECIDED, not where it is used. Parsing
// it costs ops -- MaskState::parse takes a builder and mints arithmetic for a
// dynamic bound -- so the speculative parse builds into a scratch module that
// is dropped before anything else sees it. Only "is this bound a constant"
// survives.
void PtrAnalysis::noteUnsplittableAxes(triton::AddPtrOp op) {
  unsplittableAxes.clear();
  splitCrossesLoop = false;
  auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!tensorTy)
    return;
  unsplittableAxes.assign(tensorTy.getRank(), false);

  SmallVector<Value> masks;
  SmallVector<Value> worklist{op.getResult()};
  llvm::SmallPtrSet<Operation *, 8> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (!seen.insert(user).second)
        continue;
      if (auto load = dyn_cast<triton::LoadOp>(user)) {
        if (load.getMask())
          masks.push_back(load.getMask());
      } else if (auto store = dyn_cast<triton::StoreOp>(user)) {
        if (store.getMask())
          masks.push_back(store.getMask());
      } else if (isa<tts::GetStructuredStateOp>(user)) {
        // AND A POINTER THAT CROSSES AN scf BOUNDARY MAY NOT SPLIT AT ALL.
        // get_structured_state counts its results from the TYPE -- the loop's
        // iter-args are built before any state exists -- so a split state has
        // more components than the op has results, and the rewrite of the loop
        // is already committed by the time anything notices.
        splitCrossesLoop = true;
      } else if (isa<triton::AddPtrOp, triton::BroadcastOp,
                     triton::ExpandDimsOp>(user)) {
        // The pointer travels; the access that reads it is what carries a mask.
        worklist.push_back(user->getResult(0));
      }
    }
  }
  if (masks.empty())
    return;

  OwningOpRef<ModuleOp> scratch = ModuleOp::create(op.getLoc());
  OpBuilder scratchBuilder(scratch->getBodyRegion());
  for (Value mask : masks) {
    mlir::triton::MaskState mstate(useUnsafeMaskForRun);
    if (mstate.parse(mask, op.getLoc(), scratchBuilder).failed())
      continue;
    for (uint64_t i = 0; i < unsplittableAxes.size(); i++) {
      if (i < mstate.dims.size() && !getConstantIntValue(mstate.dims[i]))
        unsplittableAxes[i] = true;
      if (mstate.hasLows() && i < mstate.lows.size() &&
          !getConstantIntValue(mstate.lows[i]))
        unsplittableAxes[i] = true;
    }
  }
}

LogicalResult PtrAnalysis::rewriteAddptrOp(triton::AddPtrOp op) {
  OpBuilder builder(op);

  noteUnsplittableAxes(op);
  llvm::scope_exit clearAxes([&] { unsplittableAxes.clear(); });

  PtrState state;
  if (visitOperandAddptr(op, state, op.getLoc(), builder).failed()) {
    // The visit named it if it could; leaving lastRefusal alone lets the
    // deepest name reach the access, which is what rewriteOp parks.
    return failure();
  }


  knownPtrs[op.getResult()] = state;

  if (isa<RankedTensorType>(op.getPtr().getType())) {
    if (state.isStructured()) {
      auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc(),
                                                       op.getType());
      ptrMap.map(op.getResult(), maketptrOp.getResult());
    } else if (enableMakeGatherScatterTensorPtr && !state.flatShape) {
      PtrState unstructuredState;
      // Switch to unstructured state analysis to create offsets and strides
      // for the non-structured dimension.
      // NOTE: this is the only place where we switch to unstructured state
      // analysis.
      isAnalysisingUnstructured = true;
      // Visit the operand again to calculate the offsets and strides for the
      // unstructured state.
      LogicalResult result =
          visitOperandAddptr(op, unstructuredState, op.getLoc(), builder);
      // Switch back to structured state analysis.
      isAnalysisingUnstructured = false;
      if (result.failed()) {
        return refuse(
            op, "Failed to analyze ptr of tt.addptr for unstructured state");
      }
      if (state.mergeUnstructuredState(unstructuredState, op, &mergeRefusal)
              .failed()) {
        if (mergeRefusal.why)
          lastRefusal = mergeRefusal.why;
        return refuse(op, "Failed to merge unstructured state for tt.addptr");
      }
      // The 1-D index table is the gather pointer's form, not the analysis's: it is
      // built from a copy, and knownPtrs keeps the merged state with its table at
      // the tile's rank -- a later expand_dims, broadcast or addptr of this pointer
      // analyses with it.
      PtrState forPtr = state;
      forPtr.collapseGatherOffsetTo1D(builder, op.getLoc());
      demotionCounts.mergedIntoGather++;
      auto maketptrOp =
          forPtr.createTTSMakeGatherScatterTensorPtrOp(builder, op.getLoc());
      knownPtrs[op.getResult()] = state;
      ptrMap.map(op.getResult(), maketptrOp.getResult());
    } else {
      // THE STATE IS NOT STRUCTURED AND THE GATHER ARM IS CLOSED -- either the
      // flag is off, or a sum wrap makes the index table meaningless.
      lastRefusal = state.flatShape
                        ? "the state is unstructured under a sum wrap"
                        : "the state is unstructured and gather/scatter is off";
      return failure();
    }
  } else {
    // record the ptr as we have visited and built up the state for this scalar
    // pointer, which may be used by rewriteForOp later.
    ptrMap.map(op.getResult(), op.getResult());
  }
  return success();
}

// Only a pointer broadcast: an integer one is an offset some addptr consumes,
// and it is handled there. A broadcast adds no offset, so a structured source
// stays structured and there is no gather arm to mirror.
LogicalResult PtrAnalysis::rewriteStride0PtrOp(Operation *op) {
  // OFFERED EVERY OP, and these three declines are the whole set: not a pointer
  // tensor, visitOperand cannot read it, the state is not structured. None of
  // what survives adds an offset, so a structured source stays structured.
  auto res = op->getResult(0);
  auto resType = dyn_cast<RankedTensorType>(res.getType());
  if (!resType || !isa<triton::PointerType>(resType.getElementType())) {
    return failure();
  }

  OpBuilder builder(op);
  PtrState state;
  if (visitOperand(res, state, op->getLoc(), builder).failed()) {
    return failure();
  }


  knownPtrs[res] = state;

  if (!state.isStructured()) {
    return refuse(op, "stride-0 op on an unstructured pointer, leaving it");
  }

  auto maketptrOp =
      state.createTTSMakeTensorPtrOp(builder, op->getLoc(), res.getType());
  ptrMap.map(res, maketptrOp.getResult());
  return success();
}

// A structured source is re-made in the conversion's layout like any stride-0 op
// (visitOperand sees through the conversion). A gathered one is what that path
// refuses, and it needs nothing re-made: the conversion changes no shape and moves
// no address, so the access behind it takes the same gather pointer.
LogicalResult
PtrAnalysis::rewriteConvertLayoutOp(triton::gpu::ConvertLayoutOp op) {
  Value src = op.getSrc();
  auto known = knownPtrs.find(src);
  if (known == knownPtrs.end() || known->second.isStructured())
    return rewriteStride0PtrOp(op);
  Value tptr = ptrMap.lookupOrNull(src);
  if (!tptr || !triton::isPtrTypeLike(op.getType()))
    return failure();
  knownPtrs[op.getResult()] = known->second;
  ptrMap.map(op.getResult(), tptr);
  return success();
}

static bool isPointerType(Type t) {
  if (auto tensor = llvm::dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensor.getElementType());
  }
  return isa<triton::PointerType>(t);
}

FailureOr<PtrState> PtrAnalysis::getLoopInitArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto ptr = forOp.getInitArgs()[index];

  // If the pointer into the scf.for was defined by tts.get_structured_state,
  // we can get the pointer state from the original pointer (the op's input):
  //
  // %ptr, %offset_1, %offset_2,..., %stride_1, %stride_2,... =
  // tts.get_structured_state %original
  // scf.for ... (%ptr) {...}
  if (auto getStateOp = ptr.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto originalPtr = getStateOp->getOperand(0);
    if (knownPtrs.count(originalPtr)) {
      return knownPtrs[originalPtr];
    }
  }

  // For nested loops scenarios, a pointer in init-args can be returned from
  // another loop of the same level:
  // e.g.:
  // clang-format off
  //  %22:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //    %23 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5) -> (tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      scf.yield %26 : tensor<2x2x!tt.ptr<f32>>
  //    }
  //    %24:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %23, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
  //      %27 = tt.addptr %arg8, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      ...
  //    }
  //    ...
  //  }
  // clang-format on
  // Notice %arg8 = %23 comes from the return value of the first loop.
  if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    return getLoopResultPtrState(forOp, index);
  }

  // If the pointer isn't defined by tts.get_structured_state nor another loop,
  // it means the current pointer is an iterarg of the outer loop.
  // In such cases, the outer loops would have already set up the PtrState for
  // us already.
  //
  // scf.for iterargs(%ptr = %init_arg) {
  //    scf.for iterargs(%ptr1 = %ptr) {  <--- we're dealing with `%ptr1` here.
  //          ...
  //    }
  // }
  if (knownPtrs.count(ptr)) {
    assert(!ptr.getDefiningOp() && "Expect the ptr to be an iterarg");
    return knownPtrs[ptr];
  }

  return failure();
}

PtrState PtrAnalysis::reconcileLoopPtrState(
    scf::ForOp forOp, size_t iterArgIndex, const PtrState &state,
    llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal) {
  PtrState newState = state;
  int cnt = iterArgIndex + 1;
  if (newState.getRank() == 0) {
    assert(newState.scalar);
    // for scalar pointers, the scalar contains the offset and is the only
    // relevant newState that could be updated by the loop.
    newState.scalar = getReplacementVal(forOp, cnt);
  } else {
    for (auto &a : newState.axes)
      a.offset = getReplacementVal(forOp, cnt++);

    for (auto &a : newState.axes)
      a.stride = getReplacementVal(forOp, cnt++);

    // THE THIRD COMPONENT, in the order rewriteGetStructuredStateOp pushed it.
    // A rank>0 state always has strides, so it always has this slot.
    newState.baseOffset = getReplacementVal(forOp, cnt++);

    // AND `scalar` IS THE SAME NUMBER, so it has to be the same iteration's.
    // It is set only where the state describes ONE value, and after the loop
    // advances that value it IS the new base offset. Left behind it names the
    // pre-loop iteration, and mulState scales the updated side by it.
    if (newState.scalar)
      newState.scalar = stateStandsStill(state)
                            ? cast<Value>(newState.baseOffset)
                            : Value();
  }

  return newState;
}

FailureOr<PtrState> PtrAnalysis::getLoopIterArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  if (!state->isStructured()) {
    // Skip if the loop init arg is not structured.
    return failure();
  }

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op.getRegionIterArg(index); });
}

FailureOr<PtrState> PtrAnalysis::getLoopResultPtrState(scf::ForOp forOp,
                                                       size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  if (!state->isStructured()) {
    // Skip if the loop init arg is not structured.
    return failure();
  }
  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op->getResult(index); });
}

LogicalResult PtrAnalysis::rewriteForOp(scf::ForOp op) {
  for (auto [i, arg] : llvm::enumerate(op.getRegionIterArgs())) {
    if (!maybeStructuredArgs.contains(arg)) {
      continue;
    }

    auto state = getLoopIterArgPtrState(op, i);
    if (failed(state)) {
      // Because the maybeStructuredArgs may contain values that are not
      // considered structured by PtrAnalysis, failing to retrieve the PtrState
      // should not fail the rewrite process.
      // We emit an error for diagnostics and debugging purposes.
      LLVM_DEBUG(op->emitWarning(
          "Rewrite for-op failed. Could not find PtrState for iter-arg index " +
          std::to_string(i)));
      continue;
    }
    // Skip when no structured dimension exists
    if (state->noStructuredDimExists())
      continue;

    // Save the current init arg's PtrState. NOT SETTLED HERE ANY MORE: the
    // base offset is a carried component now, so by this point it IS the
    // loop's block argument -- folding it onto an axis would build an
    // `arith.addi` outside the loop from a value defined inside it. The
    // make_tptr at the top of the body settles it where it is legal to.
    knownPtrs[arg] = state.value();

    // For tensors of pointers, create a tts.make_tptr at the beginning of the
    // loop body that correspond to this region iter arg. In case it is used
    // by tt.load/tt.store in the loop body before pointer updates, this will
    // make sure rewriteLoadOp/rewriteStoreOp can use the analysis result.
    // E.g., given the following input (%tensor_of_ptr is a block arg):
    // scf.for (%tensor_of_ptr) {
    //   %data = tt.load %tensor_of_ptr
    //   // more operations to update %tensor_of_ptr
    // }
    // We may produce the following output:
    // scf.for (%base_ptr, %stride, %offset) {
    //   %tensor_of_ptr = tts.make_tptr(%base_ptr, %stride, %offset)
    //   %data = tts.load %tensor_of_ptr
    //   // more operations to update %offset
    // }
    // If %tensor_of_ptr is not used (i.e., %tensor_of_ptr is updated before
    // used in the original IR), canonicalization removes it -- but ONLY while
    // it stays in one loop, as above. Handed from loop to loop (an inner loop's
    // init, or a sibling loop's, taking the previous loop's result) no single
    // scf.for ever sees an unused end and its iter-arg folder never fires. That
    // shape needs `--remove-dead-values`; see test/.../dead_ptr_iterargs.mlir.

    // For scalar pointers, there is no need to create a tts.addptr at the
    // beginning of the loop body. We don't lower tt.load and tt.store on
    // scalars in this pass; pointer arithmetics can also just use the
    // original pointer.
    // Note that there can be tensor of indices in iter-arg, so we only create
    // the make_tensor_ptr op when the arg is of pointer type.
    if (isPointerType(arg.getType())) {
      if (state->getRank() != 0) {
        OpBuilder builder(op.getRegion());
        auto maketptrOp =
            state->createTTSMakeTensorPtrOp(builder, op.getLoc(), arg.getType());
        ptrMap.map(arg, maketptrOp.getResult());
      }
    }
  }

  // Recursively rewrite the inner ops
  if (rewriteOp(op).failed()) {
    return refuse(op, "update loop body failed when rewriting for op");
  }

  return success();
}

LogicalResult
PtrAnalysis::rewriteGetStructuredStateOp(tts::GetStructuredStateOp op) {
  auto tritonValue = op->getOperand(0);

  // If this triton value isn't known, it means PtrAnalysis has failed to
  // analyze this pointer. In such cases, simply remap all uses of the
  // structured value back to its original triton value.
  if (!knownPtrs.contains(tritonValue)) {
    LLVM_DEBUG(op.emitRemark(
        "Rewrite GetStructuredStateOp failed. Could not find PtrState."));
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }

  tts::PtrState state = knownPtrs[tritonValue];
  if (!state.isStructured()) {
    LLVM_DEBUG(op.emitRemark(
        "Rewrite GetStructuredStateOp failed. PtrState is not structured."));
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }
  Value remappedValue =
      ptrMap.contains(tritonValue) ? ptrMap.lookup(tritonValue) : tritonValue;

  SmallVector<Value> replacements{remappedValue};
  OpBuilder builder(op);

  if (state.getRank() == 0) {
    // For scalar pointers, the scalar contains the offset and is the only
    // relevant state that could be updated by the loop.
    if (state.scalar) {
      replacements.push_back(state.scalar);
    } else {
      // This operand is a pointer directly from the kernel arguments.
      // Use offset 0.
      assert(!tritonValue.getDefiningOp());
      replacements.push_back(arith::ConstantOp::create(
          builder, op.getLoc(), builder.getIndexAttr(0)));
    }
  } else {
    for (auto [j, s] : llvm::enumerate(state.offsetsOf())) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = arith::ConstantOp::create(
            builder, op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }

    for (auto [j, s] : llvm::enumerate(state.stridesOf())) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = arith::ConstantOp::create(
            builder, op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }
    // THE THIRD COMPONENT, and zero when the state has none: the count comes
    // from the type, so the op always has the slot even where the pointer
    // never leaves the wrap.
    if (tts::GetStructuredStateOp::getBaseOffsetSegmentSize(
            tritonValue.getType())) {
      replacements.push_back(
          state.baseOffset
              ? getValueOrCreateConstantIndexOp(builder, op.getLoc(),
                                                state.baseOffset)
              : arith::ConstantOp::create(builder, op.getLoc(),
                                          builder.getIndexAttr(0))
                    .getResult());
    }
  }

  // A SPLIT STATE HAS MORE COMPONENTS THAN THIS OP HAS RESULTS. The result
  // count comes from the TYPE -- one offset and one stride per TENSOR axis --
  // and splitting an axis grows the STATE's rank alone, so a pointer that
  // crosses an scf boundary with a split axis arrives here with a mismatch.
  // Refusing is the whole answer available today: rewriteForOp reads these
  // results positionally, so handing it a different count is not a repair.
  // This used to reach replaceAllUsesWith and abort inside MLIR.
  if (replacements.size() != op->getNumResults()) {
    LLVM_DEBUG(op.emitRemark("Rewrite GetStructuredStateOp failed. The state "
                             "has more components than the op has results, "
                             "which is a split axis crossing this boundary."));
    lastRefusal = "a split axis crosses a loop boundary, and "
                  "get_structured_state counts its results by the tensor";
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }
  op->replaceAllUsesWith(replacements);
  op->erase();
  return success();
}

// The mask's per-axis lower bounds as tts.load/store carry them. A LOW IS A BOX
// END LIKE A DIM and those ops now take it static OR dynamic, so nothing here
// has to be a constant -- a padded convolution's `i - PAD + h` is a value.
static SmallVector<OpFoldResult>
maskLows(const mlir::triton::MaskState &mstate) {
  SmallVector<OpFoldResult> lows;
  if (mstate.hasLows())
    lows.assign(mstate.lows.begin(), mstate.lows.end());
  return lows;
}

// THE TILE A POINTER WALKS, whichever of the two spellings it wears:
// make_tptr is a tensor OF pointers, make_gather_scatter_tptr a pointer TO a
// tensor, and a split has to be folded back through both.
static std::optional<ArrayRef<int64_t>> tileShapeOf(Value ptr) {
  if (auto t = dyn_cast<RankedTensorType>(ptr.getType()))
    return t.getShape();
  if (auto p = dyn_cast<triton::PointerType>(ptr.getType()))
    if (auto t = dyn_cast<RankedTensorType>(p.getPointeeType()))
      return t.getShape();
  return std::nullopt;
}

// A SPLIT AXIS PARTS THE TWO RANKS, and this puts them back together. The
// grouping is derivable from the two shapes alone -- consecutive state axes
// multiply up to one tensor extent -- so neither op has to carry the state.
static std::optional<SmallVector<ReassociationIndices>>
reassociationBetween(ArrayRef<int64_t> stateShape, ArrayRef<int64_t> tensor) {
  SmallVector<ReassociationIndices> re;
  int64_t g = 0;
  for (int64_t want : tensor) {
    ReassociationIndices grp;
    int64_t prod = 1;
    do {
      if (g >= (int64_t)stateShape.size())
        return std::nullopt;
      prod *= stateShape[g];
      grp.push_back(g++);
    } while (prod < want);
    if (prod != want)
      return std::nullopt;
    re.push_back(grp);
  }
  return g == (int64_t)stateShape.size() ? std::optional(re) : std::nullopt;
}

// THE MASK IS PER TENSOR AXIS AND THE TILE IS PER STATE AXIS, and a split parts
// them. `m` masked elements of an axis cut into [c1, ..., cr] is a box exactly
// when the product of the inner extents divides m; otherwise the masked region
// is ragged across the cut and no per-axis bound says it.
static std::optional<SmallVector<OpFoldResult>>
spreadOverSplit(ArrayRef<OpFoldResult> perAxis, ArrayRef<int64_t> tile,
                ArrayRef<ReassociationIndices> re, bool isBound,
                OpBuilder &builder) {
  SmallVector<OpFoldResult> out;
  if (perAxis.size() != re.size())
    return std::nullopt;
  for (auto [t, group] : llvm::enumerate(re)) {
    if (group.size() == 1) {
      out.push_back(perAxis[t]);
      continue;
    }
    int64_t inner = 1;
    for (size_t j = 1; j < group.size(); j++)
      inner *= tile[group[j]];
    auto m = getConstantIntValue(perAxis[t]);
    if (!m || *m % inner != 0)
      return std::nullopt;
    out.push_back(builder.getIndexAttr(*m / inner));
    for (size_t j = 1; j < group.size(); j++)
      out.push_back(builder.getIndexAttr(isBound ? tile[group[j]] : 0));
  }
  return out;
}

LogicalResult PtrAnalysis::rewriteLoadOp(triton::LoadOp op,
                                         bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto mask = op.getMask();
  auto other = op.getOther();
  auto loc = op.getLoc();

  if (!ptr) {
    return refuse(op, "pointer is not replace with tts.make_tptr so loadOp "
                      "cannot be rewritten");
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    return refuse(op, "scalar loadOp will not be rewritten");
  }

  ArrayRef<OpFoldResult> dims;
  SmallVector<OpFoldResult> spreadStorage;
  SmallVector<OpFoldResult> lows;
  mlir::triton::MaskState mstate(useUnsafeMask);
  Value scalarOther;
  Value elemMask;

  OpBuilder builder(op);
  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      return refuse(op, "MaskAnalysis failed");
    }
    // WHICH AXIS THE MASK IS ON PICKS THE CARRIER, not which direction the
    // access goes. A mask on the gathered axis says WHICH ROWS PARTICIPATE,
    // which is what a partial gather IS -- an address fact, and the pointer
    // describes addresses. Anything else is per element and rides on the
    // access. Realising either -- a select, a clamp, a park -- stays the
    // consumer's call.
    foldScalarPredicate(mstate, loc, builder);
    auto unstructured = mstate.getUnstructuredMasks();
    if (!unstructured.empty()) {
      if (unstructured.size() > 1) {
        return refuse(
            op, "MaskAnalysis failed for more than one unstructured masks");
      }
      // The axis the mask owns is read WHOLE: the consumer is what narrows it,
      // and the `0` that marks a given-up axis would read nothing.
      auto tileType = dyn_cast<RankedTensorType>(op.getType());
      if (!tileType) {
        return failure();
      }
      unsigned dim = unstructured.front().first;
      if (Value gathered = rebuildGatherWithMask(op, ptr, mstate, loc, builder)) {
        ptr = gathered;
      } else {
        elemMask = mask;
        mstate.dims[dim] = builder.getIndexAttr(tileType.getShape()[dim]);
        if (mstate.hasLows())
          mstate.lows[dim] = builder.getIndexAttr(0);
      }
    }
    dims = mstate.dims;
    lows = maskLows(mstate);

    // THE TILE MAY OUTRANK THE VALUE once an axis is split, and the mask is
    // still one bound per tensor axis. Spread it over the cut, or refuse.
    if (auto tile = tileShapeOf(ptr)) {
      auto want = dyn_cast<RankedTensorType>(op.getType());
      if (want && tile->size() != want.getRank()) {
        auto re = reassociationBetween(*tile, want.getShape());
        auto spreadDims = re ? spreadOverSplit(dims, *tile, *re, true, builder)
                             : std::nullopt;
        // An absent lower bound stays absent -- maskLows returns an
        // empty vector for that, and spreading it would refuse a mask that has
        // nothing to spread.
        std::optional<SmallVector<OpFoldResult>> spreadLows;
        if (lows.empty())
          spreadLows = SmallVector<OpFoldResult>{};
        else if (re)
          spreadLows = spreadOverSplit(lows, *tile, *re, false, builder);
        if (!spreadDims || !spreadLows) {
          return refuse(op, "the mask does not survive the axis split");
        }
        spreadStorage = *spreadDims;
        dims = spreadStorage;
        lows = *spreadLows;
      }
    }
  }

  if (other) {
    assert(mask && "other value used while no masks are specified");

    scalarOther = utils::getScalarValue(other, loc, builder);
    if (!scalarOther) {
      return refuse(op, "other value used in masked load produced by "
                        "unsupported instruction");
    }
  }

  // The structured load keeps the layout of the load it replaces -- an encoding is
  // a fact about the value, not about how its address was spelled -- reshaped
  // onto the tile when a split made the two differ.
  Attribute encoding;
  if (auto tile = tileShapeOf(ptr))
    encoding = triton::encodingReplacing(op.getType(), *tile);
  auto loadOp = tts::LoadOp::create(builder, loc, ptr, dims, lows, scalarOther,
                                    elemMask, encoding);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::load:\n";
    loadOp->dump();
  });

  Value result = loadOp.getResult();
  auto want = dyn_cast<RankedTensorType>(op.getType());
  auto got = dyn_cast<RankedTensorType>(result.getType());
  if (want && got && want.getRank() != got.getRank()) {
    auto re = reassociationBetween(got.getShape(), want.getShape());
    if (!re) {
      return refuse(op, "the split tile does not fold back to the value");
    }
    result = tensor::CollapseShapeOp::create(builder, loc, want, result, *re);
  }
  op.replaceAllUsesWith(result);
  op->erase();
  return success();
}

// Structured values from the TritonStructuredDialect have offsets and strides
// that might change in each loop iteration and hence will appear in an scf.for
// iter-args like so:
//
// %structured, %offsets, %strides  = tts.get_structured_state
// scf.for (%arg0 = %structured, %arg1 = %offsets, %arg2 = %strides) {
//   %a = %arg0 + 1
//   %b = %b + 2
//   scf.for (%arg1 = %b) {
//      ...
//   }
// }
//
// In `rewriteForOp`, we have to recognize such structured values in order to
// rewrite their PtrState accordingly. Previously, only values of Pointer-like
// type (e.g.: tensor<tt.ptr<>> or tt.ptr<tensor<>>), so detecting these values
// is as easy as checking the type.
//
// Now, tensor of indices could also appear in a loop's iter-arg. To reliably
// detect all such cases, we perform a BFS-like traversal of the IR where the
// sources are the results of `tts.get_structured_state`. All values that
// originate from the results of `tts.get_structured_state` are consider
// "maybeStructured". If a loop's iter-arg is considered "maybeStructured", we
// must set up their PtrState during `rewriteForOp`.
void PtrAnalysis::initializeMaybeStructuredArgs(Operation *op) {
  std::queue<Value> q;
  DenseSet<Value> visited;

  op->walk([&q, &visited](tts::GetStructuredStateOp getStateOp) {
    Value value = getStateOp->getResult(0);
    visited.insert(value);
    q.push(value);
  });

  while (!q.empty()) {
    auto v = q.front();
    q.pop();
    for (auto user : v.getUsers()) {
      // scf.for is a special case. We have 2 set of values to consider:
      // - iter-args
      // - loop results
      // for every init arg that originates from a `tts.get_structured_state`
      // op, its corresponding iter-arg and loop result will also be considered
      // "maybeStructured".
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (auto [argIndex, arg] :
             llvm::zip(llvm::index_range(0, forOp.getInitArgs().size()),
                       forOp.getInitArgs())) {
          if (arg != v) {
            continue;
          }
          auto iterArg = forOp.getRegionIterArg(argIndex);
          auto tiedLoopRes = forOp.getTiedLoopResult(iterArg);
          SmallVector<Value> neighbors{iterArg, tiedLoopRes};
          for (auto neighbor : neighbors) {
            maybeStructuredArgs.insert(neighbor);
            if (!visited.contains(neighbor)) {
              visited.insert(neighbor);
              q.push(neighbor);
            }
          }
        }
      } else {
        for (auto res : user->getResults()) {
          if (res.getType() != v.getType()) {
            continue;
          }
          maybeStructuredArgs.insert(res);
          if (!visited.contains(res)) {
            visited.insert(res);
            q.push(res);
          }
        }
      }
    }
  }
}

LogicalResult PtrAnalysis::rewriteStoreOp(triton::StoreOp op,
                                          bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getValue();
  auto mask = op.getMask();
  auto loc = op.getLoc();

  if (!ptr) {
    return refuse(op, "pointer is not replace with tts.make_tptr so storeOp "
                      "cannot be rewritten");
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    return refuse(op, "scalar storeOp will not be rewritten");
  }

  ArrayRef<OpFoldResult> dims;
  SmallVector<OpFoldResult> spreadStorage;
  SmallVector<OpFoldResult> lows;
  Value elemMask;
  mlir::triton::MaskState mstate(useUnsafeMask);

  OpBuilder builder(op);

  // Analyze the mask operand to determine at runtime the size of the data
  // are moving.
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      return refuse(op, "MaskAnalysis failed");
    }
    // THE SAME RULE AS THE LOAD, which is the point: the axis picks the carrier
    // and the direction does not. A mask on the gathered axis is the partial
    // gather itself and goes on the pointer; anything else is per element and
    // goes in tts.store's own slot, where nothing about the pointer is lost.
    foldScalarPredicate(mstate, loc, builder);
    if (!mstate.getUnstructuredMasks().empty()) {
      // THE WHOLE MASK, not the axis fragment MaskState kept: elem_mask is one
      // bit per ELEMENT, and triton already broadcast the operand to the tile.
      // The box dims stay as they are -- a clamp and a mask that agree are the
      // same answer, and the clamp is what sizes the transfer.
      auto [dim, m] = mstate.getUnstructuredMasks()[0];
      (void)m;
      if (Value gathered = rebuildGatherWithMask(op, ptr, mstate, loc, builder)) {
        ptr = gathered;
      } else {
        elemMask = mask;
        mstate.dims[dim] = builder.getIndexAttr(
            cast<ShapedType>(val.getType()).getShape()[dim]);
        mstate.masks[dim] = nullptr;
        if (mstate.hasLows())
          mstate.lows[dim] = builder.getIndexAttr(0);
      }
    }
    dims = mstate.dims;
    lows = maskLows(mstate);
    // THE SAME SPREAD AS THE LOAD: one bound per tensor axis has to become one
    // per state axis once a split parts the two.
    if (auto tile = tileShapeOf(ptr)) {
      auto have = dyn_cast<RankedTensorType>(val.getType());
      if (have && have.getRank() != (int64_t)tile->size()) {
        auto re = reassociationBetween(*tile, have.getShape());
        auto spreadDims = re ? spreadOverSplit(dims, *tile, *re, true, builder)
                             : std::nullopt;
        std::optional<SmallVector<OpFoldResult>> spreadLows;
        if (lows.empty())
          spreadLows = SmallVector<OpFoldResult>{};
        else if (re)
          spreadLows = spreadOverSplit(lows, *tile, *re, false, builder);
        if (!spreadDims || !spreadLows) {
          return refuse(op, "the mask does not survive the axis split");
        }
        spreadStorage = *spreadDims;
        dims = spreadStorage;
        lows = *spreadLows;
      }
    }
  }

  // THE MIRROR OF THE LOAD'S FOLD: the pointer may walk more axes than the
  // value has, so the value is expanded to meet it.
  if (auto tile = tileShapeOf(ptr)) {
    auto have = dyn_cast<RankedTensorType>(val.getType());
    if (have && have.getRank() != (int64_t)tile->size()) {
      auto re = reassociationBetween(*tile, have.getShape());
      if (!re) {
        return refuse(op, "the value does not fold onto the split tile");
      }
      val = tensor::ExpandShapeOp::create(
                builder, loc,
                RankedTensorType::get(*tile, have.getElementType(),
                                      triton::reshapeEncoding(have.getShape(),
                                                              have.getEncoding(), *tile)),
                val, *re)
                .getResult();
    }
  }

  auto storeOp =
      tts::StoreOp::create(builder, loc, ptr, val, dims, lows, elemMask);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::store:\n";
    storeOp->dump();
  });

  op->erase();
  return success();
}

static std::optional<ArrayRef<int64_t>> tileShapeOf(Value ptr);

// THE ONLY COMPLETE OBSERVATION POINT. 107 places return a bare failure and
// none of them knows it ended an access; this one does. Loads and stores only --
// a failed addptr takes its load with it, so counting both counts one twice.
// WHICH RUNG THE ACCESS ACTUALLY LANDED ON, read off the pointer it was given.
// Captured before the rewrite because that erases the access.
void PtrAnalysis::recordRung(Value ptr, Value tptr) {
  if (!tileShapeOf(ptr))
    return;
  if (tptr && isa_and_nonnull<tts::MakeGatherScatterTensorPtrOp>(
                  tptr.getDefiningOp()))
    demotionCounts.rung2++;
  else
    demotionCounts.rung1++;
}

void PtrAnalysis::recordNotRewritten(Value ptr) {
  if (!tileShapeOf(ptr)) {
    demotionCounts.neverRewrittenScalar++;
    return;
  }
  auto it = refusalOfPtr.find(ptr);
  const char *why = it != refusalOfPtr.end() ? it->second : lastRefusal;
  demotionCounts.neverRewritten++;
  demotionCounts.neverRewrittenReasons[why ? why : "not named here"]++;
}

LogicalResult PtrAnalysis::rewriteOp(Operation *rootOp, bool useUnsafeMask) {
  useUnsafeMaskForRun = useUnsafeMask;
  LLVM_DEBUG({
    llvm::dbgs() << "rewriting rootOp\n";
    rootOp->dump();
  });

  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == rootOp) {
      return WalkResult::advance();
    }
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<triton::AddPtrOp>([&](auto addptr) {
          lastRefusal = nullptr;
          if (rewriteAddptrOp(addptr).failed()) {
            LLVM_DEBUG(
                addptr->emitRemark("PtrAnalysis: Failed to rewrite AddPtrOp"));
            // Parked, not counted: the ACCESS is what gets counted, below.
            if (lastRefusal)
              refusalOfPtr[addptr.getResult()] = lastRefusal;
          }
          return WalkResult::advance();
        })
        .Case<triton::gpu::ConvertLayoutOp>([&](auto cvt) {
          if (rewriteConvertLayoutOp(cvt).failed()) {
            LLVM_DEBUG(cvt->emitRemark(
                "PtrAnalysis: Failed to rewrite ConvertLayoutOp"));
          }
          return WalkResult::advance();
        })
        .Case<triton::LoadOp>([&](auto load) {
          lastRefusal = nullptr;
          Value ptr = load.getPtr(), tptr = ptrMap.lookupOrNull(ptr);
          if (rewriteLoadOp(load, useUnsafeMask).failed()) {
            LLVM_DEBUG(
                load->emitRemark("PtrAnalysis: Failed to rewrite LoadOp"));
            recordNotRewritten(ptr);
            return WalkResult::advance();
          }
          recordRung(ptr, tptr);
          return WalkResult::skip();
        })
        .Case<triton::StoreOp>([&](auto store) {
          lastRefusal = nullptr;
          Value ptr = store.getPtr(), tptr = ptrMap.lookupOrNull(ptr);
          if (rewriteStoreOp(store, useUnsafeMask).failed()) {
            LLVM_DEBUG(
                store->emitRemark("PtrAnalysis: Failed to rewrite StoreOp"));
            recordNotRewritten(ptr);
            return WalkResult::advance();
          }
          recordRung(ptr, tptr);
          return WalkResult::skip();
        })
        .Case<scf::ForOp>([&](auto forOp) {
          // `rewriteForOp` recursively visits its children, so regardless
          // whether the rewrite succeeds or not, we need to return "skip" so
          // that the the walk does not visit the for-op's child operations
          // the second time.
          if (rewriteForOp(forOp).failed()) {
            LLVM_DEBUG(
                forOp->emitRemark("PtrAnalysis: Failed to rewrite ForOp"));
          }
          return WalkResult::skip();
        })
        .Case<tts::GetStructuredStateOp>(
            [&](tts::GetStructuredStateOp getStateOp) {
              // For tensor of indices potentially being used in pointer
              // arithmetic sequence, we need to manually populate the state of
              // none already exists.
              // This process is necessary because unlike triton pointers in a
              // loop which always have a `tt.addptr` that triggers the rewrite
              // process which includes generating the ops for updating offsets
              // and strides, tensor of indices only have a simple `arith.addi`
              // (or other arith ops).
              // Without visiting these ops manually, the ops to update the
              // offsets and strides would not be generated.
              auto tritonValue = getStateOp->getOperand(0);
              if (!knownPtrs.contains(tritonValue)) {
                PtrState state;
                OpBuilder b(getStateOp);
                if (succeeded(visitOperand(tritonValue, state,
                                           getStateOp->getLoc(), b)) &&
                    state.isStructured()) {
                  knownPtrs[tritonValue] = state;
                } else {
                  LLVM_DEBUG(getStateOp->emitRemark(
                      "PtrAnalysis: Failed to populate ptr "
                      "state for tensor of indices"));
                }
              }

              return WalkResult::skip();
            })
        .Default([&](Operation *other) {
          // NO LIST OF OP NAMES HERE. Which ops are stride-0 pointer ops is a
          // question visitOperand already answers; naming them a second time
          // is a copy that drifts (philosophy 13, rule 5).
          //
          // WHAT BOUNDS OFFERING IT TO EVERYTHING is visitOperand itself, not a
          // measurement: rewriteStride0PtrOp declines a result that is not a
          // ranked tensor of pointers, then declines whatever visitOperand
          // cannot read. Its tensor arm is a closed else-if chain, and the only
          // pointer-tensor producers in it are the four that used to be listed
          // plus tt.addptr, which has its own Case. tt.int_to_ptr and tt.cat
          // reach this Default and are refused there -- their visitOperand arms
          // are in the SCALAR pointer branch, which a tensor never enters.
          if (other->getNumResults() >= 1) {
            (void)rewriteStride0PtrOp(other);
          }
          return WalkResult::advance();
        });
  });

  return success();
}

} // namespace tts
} // namespace mlir
