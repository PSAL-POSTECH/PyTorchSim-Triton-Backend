//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSISSTRUCTURED_PTRANALYSIS_H
#define TRITON_ANALYSISSTRUCTURED_PTRANALYSIS_H

#include <map>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "PyTorchSimExt/Analysis/MaskAnalysis.h"
#include "PyTorchSimExt/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <cstddef>
#include <set>

namespace mlir {

class OpBuilder;

namespace tts {

const extern std::string ptrAnalysisAttr;

// WHY A MERGE REFUSED, and whether the step slot is what did it. The name is
// for the report; `fromStep` is what visitOperandAddptr's retry keys on, and the
// two must stay separable -- retrying a refusal the step did not cause would
// change answers, not just cost time. At namespace scope because PtrState fills
// it in and PtrAnalysis reads it.
struct Refusal {
  const char *why = nullptr;
  bool fromStep = false;
};

// Data structure used to decode pointer arithmetics. offsets, sizes, and
// strides are in unit of elements in a linearly laid-out memory, which is the
// same as pointer arithmetic operations in Triton language. scalar is a
// shortcut used when the entire state describes a single scalar value. source
// is the base pointer. If order is present, PtrState describes block pointer;
// otherwise it describes non-block pointers. When it describes block pointer,
// shape field means the same field as tt.make_tensor_ptr; when it describes a
// non-block pointer, shape field indicates how address wraps around (i.e.,
// modulo); a constant 0 indicates no modulo for the dimension.
// Multi-dimension PtrState, which has one unstructured dimension, is supported
// for gather/scatter access. The unstructured dimension is marked by a tensor
// type offset. The tensor offset for the unstructured dimension must be
// expanded from a 1D tensor. The analysis will fail for multi-dimension
// unstructured offsets. Later, when using the tensor offset to calculate the
// address, it will be collapsed to 1D. To support gather/scatter access, treat
// the unstructured offset as a whole offset instead of decoding the pointer
// arithmetic on it except scalar mul.
// The stride is set to 1 when there's no scalar mul so it still matches the
// offset * stride formula. When there're scalar muls, the stride is set to the
// multiplication of all the scalar strides.
struct PtrState {
  // ONE STATE AXIS. These were five vectors that had to stay the same length,
  // and the only thing saying so was an assert this build compiles out. A
  // vector of these cannot come apart.
  struct Axis {
    OpFoldResult offset;
    OpFoldResult size;
    OpFoldResult stride;
    // THE WRAP: this axis reads `(offset + p*stride) mod shape`. A constant 0
    // is no wrap, which is what almost every axis holds.
    OpFoldResult shape;
    // THE PERIOD: this axis's index is `p // step`. 0 is no step. `p // K` is
    // constant on runs of K, and when K does not divide the extent the runs are
    // ragged -- 32 // 11 is 11, 11, 10 -- so the period is what is carried and
    // the consumer walks the runs off it.
    int64_t step = 0;
  };
  SmallVector<Axis> axes;

  // BLOCK POINTERS ONLY, and not part of the group above: it arrives whole from
  // tt.make_tensor_ptr and is empty for everything else, which is what
  // isBlockPtr() reads.
  SmallVector<int32_t> order;

  // HOW MANY STATE AXES EACH TENSOR AXIS OCCUPIES. `p // K` cuts one axis into
  // two, so the state can outrank the tensor and the load folds it back with a
  // collapse_shape. EMPTY MEANS ONE EACH -- the common case, and what lets
  // everything that builds or reshapes a state ignore this; only code that cuts
  // an axis or reads across one has to maintain it.
  SmallVector<int32_t> split;


  Value source;
  Value scalar;

  // An address term that lives OUTSIDE every wrap: what a splat adds to the
  // whole window. Inside a wrap is a different fact in a different slot --
  // offsets[i], which is the value the modulo arithmetic reads.
  OpFoldResult baseOffset;

  // A modulus on the SUM of every axis, taken after each axis's own wrap.
  // `shape` is per-axis and keeps the other axes affine; this one belongs to no
  // axis and makes every axis non-affine -- which is a fact worth carrying, not
  // a reason to drop the pointer a rung.
  OpFoldResult flatShape;

  // THE WIDTH A TENSOR OFFSET OF THIS STATE STANDS IN. A structured offset is
  // an Index scalar and has no width; only an offset that is already a tensor
  // does, and every such offset in one state has to agree.
  Type tensorOffsetType(OpBuilder &builder) const;

  // Raise that width and bring the offsets already in the state up to it.
  void widenOffsets(Type elemTy, OpBuilder &builder, Location loc);

  // THE STATE'S RANK, which is what the tptr and the DMA descriptor walk.
  // THE COLUMNS, FOR THE PLACES THAT STILL WANT THEM. Building an op takes one
  // array per field and MLIR's own accessors hand them back the same way, so
  // the four are cut out here rather than at every call.
  SmallVector<OpFoldResult> offsetsOf() const;
  SmallVector<OpFoldResult> sizesOf() const;
  SmallVector<OpFoldResult> stridesOf() const;
  SmallVector<OpFoldResult> shapeOf() const;
  SmallVector<int64_t> stepsOf() const;
  void setAxes(ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
               ArrayRef<OpFoldResult> strides, ArrayRef<OpFoldResult> shape);

  int32_t getRank() const;

  // The TENSOR's rank -- what the value's type, the mask box and every shape op
  // count in. Equal to getRank() until an axis is split.
  int32_t tensorRank() const;

  // Tensor axis `t`'s state axes, as [begin, end). Outermost first, so a split
  // of `p = outer*K + inner` puts outer at begin and inner at begin + 1.
  std::pair<int32_t, int32_t> axisRange(int32_t t) const;

  // WHAT BOTH CREATORS BELOW ASK OF THE STATE, ASKED ONCE. Each used to open
  // with the same six-line loop over `sizes` and the same four-line read of
  // `baseOffset`; the two copies are why an experiment that blocked one
  // creator's base offset left the other one's untouched.
  SmallVector<int64_t> getStaticSizes() const;

  // The base offset as a Value for the op's own `base_offset` slot, or null
  // when the state has none. It belongs to NO axis, so it is never folded onto
  // one here -- see the slot's comment in the .td and getCarriedBaseOffset's
  // body for the two places upstream that legitimately do fold one.
  Value getCarriedBaseOffset(OpBuilder &builder, Location loc) const;

  bool isEmpty() const;

  bool hasModulo() const;

  // Axis d's step, or 0. Reads an empty `step` as all-zero.
  int64_t stepOf(uint32_t d) const;
  bool hasStep() const;

  bool dimHasModulo(uint32_t dim) const;

  bool dimIsStructured(uint32_t dim) const;
  int32_t getNonStructuredDim() const;
  // Verify that all dimensions are not structured.
  bool noStructuredDimExists() const;

  bool isStructured() const;

  bool isBlockPtr() const;

  void dump() const;

  // For unsupported op, save the op to the state.
  LogicalResult rebuildAsUnsupportedOp(Value op);

  // When merge with other state which is not structured, set the nonContinuous
  // dimension offset as op.
  // Fail if the operation already mixes different dimensions.
  // For case
  // clang-format off
  //    %14 = tt.expand_dims %11 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  //    %dim0_value = tt.broadcast %14 : tensor<64x1xi1> -> tensor<64x64xi1>
  //    %16 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
  //    %dim1_value = tt.broadcast %16 : tensor<1x64xi1> -> tensor<64x64xi1>
  //    add  %dim0_value, %dim1_value
  // clang-format on
  //    the add will have size > 1 for both dim0 and dim1.
  //    It will fail for mix of different dims.
  //
  // Fail if the operation does not contribute to nonContinuousDim.
  // For case
  // clang-format off
  //    %14 = tt.expand_dims %11 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  //    %dim0_value = tt.broadcast %14 : tensor<64x1xi1> -> tensor<64x64xi1>
  //    %16 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  //    %dim0_value2 = tt.broadcast %16 : tensor<64x1xi1> -> tensor<64x64xi1>
  //    add  %dim0_value, %dim0_value2
  // clang-format on
  //    the add only have size > 1 for dim0 which doesn't mix of different
  //    dims.
  // But if call rebuildAsGatherScatter on the add with nonContinuousDim = 1 it
  // will fail because it only have dim0.
  LogicalResult rebuildAsGatherScatter(Value op, int nonContinuousDim);

  // Line both sides up on one set of axes before a merge reads them, and fill
  // in this state's own `split` and `step` from them. addState and mulState ask
  // exactly the same thing.
  LogicalResult alignForMerge(const PtrState &lhsIn, const PtrState &rhsIn,
                              PtrState &lhs, PtrState &rhs, Operation *op,
                              OpBuilder &builder, Refusal *refusal);

  // Both sides contribute to one axis and one of them is data: fold the term
  // into a single index tensor and leave the stride at 1.
  LogicalResult foldDimOntoOneIndex(const PtrState &lhsState,
                                    const PtrState &rhsState,
                                    ArrayRef<OpFoldResult> lhsOffsets,
                                    uint64_t i,
                                    SmallVectorImpl<OpFoldResult> &offsets,
                                    SmallVectorImpl<OpFoldResult> &strides,
                                    Operation *op, OpBuilder &builder,
                                    Refusal *refusal);

  // Merge ONE axis on which one side's index is data rather than a stride.
  // Pushes that axis's offset and stride; addState's own loop keeps the case
  // where both sides are structured, which is a term-by-term add.
  LogicalResult mergeDimWithUnstructured(
      const PtrState &lhsState, const PtrState &rhsState,
      ArrayRef<OpFoldResult> lhsOffsets, uint64_t i,
      SmallVectorImpl<OpFoldResult> &offsets,
      SmallVectorImpl<OpFoldResult> &strides, bool isAnalysisingUnstructured,
      Operation *op, OpBuilder &builder, Refusal *refusal);

  // Process addition of two PtrStates. `refusedForStep`, when given, is raised
  // if the refusal was a step guard's -- the caller un-mints the step and tries
  // again rather than losing the whole address.
  LogicalResult addState(const PtrState &lhsState, const PtrState &rhsState,
                         bool isAnalysisingUnstructured, Operation *op,
                         OpBuilder &builder,
                         Refusal *refusal = nullptr);

  // Process multiplication of two PtrStates
  LogicalResult mulState(const PtrState &lhsState, const PtrState &rhsState,
                         bool isAnalysisingUnstructured, Operation *op,
                         OpBuilder &builder,
                         Refusal *refusal = nullptr);

  LogicalResult mergeUnstructuredState(const PtrState &other, Operation *op,
                                       Refusal *refusal = nullptr);

  // THE GATHER INDEX RIDES ONE AXIS, so a multi-axis offset tensor has to be
  // 1-D before it can be that axis's offset. It mutates the state on purpose:
  // the caller records the state afterwards and both have to say one shape.
  void collapseGatherOffsetTo1D(OpBuilder &builder, Location loc);

  // BOTH CREATORS READ THE STATE AND LEAVE IT ALONE. They still write IR (a
  // base offset can need a constant), but the state a caller hands in is the
  // state it keeps -- collapseGatherOffsetTo1D above is the one exception.
  // `replaces` is the type of the pointer the op stands for: its layout carries over.
  tts::MakeTensorPtrOp createTTSMakeTensorPtrOp(OpBuilder &builder, Location loc,
                                                Type replaces = {}) const;
  tts::MakeGatherScatterTensorPtrOp
  createTTSMakeGatherScatterTensorPtrOp(OpBuilder &builder, Location loc) const;
};

class PtrAnalysis {
public:
  // WHERE THE LADDER WAS CLIMBED DOWN, and by which arithmetic. A `%`, a `//`
  // or a `-` that will not fold into a stride leaves this analysis THREE ways
  // out: rebuild the value as a gather/scatter pointer, fail that rebuild, or
  // never reach one at all -- a refusal deeper in the visit, which returns a
  // bare failure and leaves the access for --triton-to-unstructured. All are
  // silent, and neither LLVM_DEBUG nor llvm::Statistic can be asked in a
  // release build -- so a caller wanting to know what stopped a kernel has
  // nothing to read. `--triton-to-structured=report-demotions` prints this.
  struct Demotions {
    // THE LADDER, IN ACCESSES -- one count per tt.load / tt.store, which is the
    // unit the rungs are about. Everything below this is in VALUES: one index
    // table absorbs however many rebuilt values fed it, so the two must not be
    // added. Measured on 448 kernels, the value counts run ~40% above these.
    //   rung 1  tts.make_tptr                    one stride for the whole tile
    //   rung 2  tts.make_gather_scatter_tptr     one axis is an index table
    //   rung 3  left as tt.load / tt.store       one address per element
    unsigned rung1 = 0, rung2 = 0;
    // Rebuilt as a gather/scatter pointer: one address per gathered row.
    unsigned rem = 0, div = 0, sub = 0, chosen = 0, other = 0;
    // Not rebuildable either: the value leaves the structured world.
    unsigned refusedRem = 0, refusedDiv = 0, refusedSub = 0,
             refusedChosen = 0, refusedOther = 0;
    // AN ADDRESS THAT DEPENDS ON DATA IS NOT A DEMOTION. No stride could ever
    // have expressed it -- that is what a gather IS -- so rung 2 is the right
    // answer and counting it beside the others inflates them. Asked of the
    // whole chain, not of one op: a coordinate is loaded, then scaled, floored
    // and converted, and only the load says where it came from.
    unsigned fromMemory = 0, refusedFromMemory = 0;
    std::map<std::string, unsigned> memoryOps, refusedMemoryOps;
    // NOR IS AN ADDRESS WITH MORE THAN ONE BASE. `base + strides` names exactly
    // one buffer, and two pointer roots reaching one value -- a `where` between
    // them, say -- cannot be folded into one whatever the strides are. There is
    // no load in that chain, so the data question above does not catch it.
    unsigned manyBases = 0, refusedManyBases = 0;
    std::map<std::string, unsigned> baseOps, refusedBaseOps;
    // WHY A DIVISION WAS NOT SPLIT. The count says how big the class is; the
    // reason says what to build next, and this build carries no LLVM_DEBUG.
    std::map<std::string, unsigned> divReasons;
    // THE ACCESSES THAT WERE NEVER REWRITTEN, counted where the walker gives up
    // on them rather than where the refusal happened -- that is the only point
    // that is complete. Loads and stores only: an addptr that fails takes its
    // load down with it, and counting both would count one access twice.
    // Without this row a refusal inside addState produced a gather and a report
    // saying "total 0", which is what these numbers are for.
    unsigned neverRewritten = 0;
    std::map<std::string, unsigned> neverRewrittenReasons;
    // A SCALAR ACCESS IS NOT A DEMOTION, for the same reason a data-dependent
    // address is not: there is no tile to structure, so no rung was lost. It is
    // counted apart because it is a large share of what is left unrewritten --
    // 35 of one kernel's 36 -- and folding it in would inflate every total.
    unsigned neverRewrittenScalar = 0;
    // THE OP OF EVERY DEMOTED VALUE, whatever the cause -- not only the ones
    // with no handler, which is what this said while collecting all of them.
    // Read it beside the cause counts: the same arith.minsi appears here and
    // under `chosen`, and that pair is the statement.
    std::map<std::string, unsigned> otherOps, refusedOtherOps;
    // THE ADDRESS AS A WHOLE WOULD NOT FOLD, so the unstructured re-analysis
    // gave one axis an index table. NOT IN valuesRebuilt(): no single
    // arithmetic asked for this, which is why an address built only out of
    // arange could reach rung 2 under a report saying nothing climbed down.
    unsigned mergedIntoGather = 0;
    // `(x / K) * K + (x % K)` read back as `x`. NOT A DEMOTION: the one
    // counter here that says something climbed UP -- an address that would
    // otherwise carry a division and a modulo becomes one affine index.
    unsigned reflattenedDivRem = 0;
    // `x // K` cut into rows of g = gcd(K, what divides the offset): one index
    // per row instead of one per element. A climb UP from a per-element gather.
    unsigned divIntoRows = 0;
    // `x % W` dropped because every access reading it is masked to `x < W`.
    // Also a climb UP: the wrap it would have become is a walk in pieces, and
    // an unreadable one is a gather over an index the access did not need.
    unsigned maskedModDropped = 0;
    // THE VALUES THIS ANALYSIS CLIMBED DOWN FOR, not the accesses that ended up
    // there -- see the note on rung1. `fromMemory` and `manyBases` are in it
    // because they are values it gave up decoding, even though the rung they
    // land on is the right one for them.
    unsigned valuesRebuilt() const {
      return rem + div + sub + chosen + other + refusedRem + refusedDiv +
             refusedSub + refusedChosen + refusedOther + fromMemory +
             refusedFromMemory + manyBases + refusedManyBases;
    }
  };
  const Demotions &demotions() const { return demotionCounts; }

private:
  Demotions demotionCounts;

  // TENSOR AXES THE ACCESS'S MASK CANNOT FOLLOW ACROSS A CUT. `static_mask_dims`
  // is a box, one bound per axis, so a bound that is not a constant cannot be
  // spread over a split: `m` elements of an axis cut into [outer, inner] is a
  // box only when inner divides m, and nobody knows m here. Read from the
  // ACCESS before its address is built, which is the one place both are in
  // reach -- the walk rewrites the addptr first, but the load is already there.
  llvm::SmallVector<bool> unsplittableAxes;
  // AND WHAT A ROW CUT MAY STILL DO WITH SUCH A BOUND: the largest number known
  // to divide every mask bound on the axis, 0 where there is no mask. A cut
  // into rows of g keeps the box when g divides it, constant or not.
  llvm::SmallVector<int64_t> maskGrain;
  // THE WHOLE POINTER, not one axis: a split state cannot cross an scf
  // boundary at all, so this is a flag rather than an entry per axis.
  bool splitCrossesLoop = false;
  void noteUnsplittableAxes(triton::AddPtrOp op);
  // What rewriteOp was asked for; the look-ahead has to read a mask the same
  // way the access will.
  bool useUnsafeMaskForRun = false;

  // `unmintDivision` is set only by visitOperandAddptr's retry, and it takes
  // BOTH of `x // K`'s slots away -- the step and the split -- because a merge
  // that could not take one cannot take the other either. `mergeRefusal` is
  // what the last addState / mulState / mergeUnstructuredState left behind: its
  // name feeds the report and its `fromStep` decides whether the retry runs.
  bool unmintDivision = false;
  Refusal mergeRefusal;

  // ONE PLACE, ONE STRING. The remark and the report were written separately at
  // every refusal, so a typo split them -- and report-demotions counts BY that
  // string, which makes the two disagreeing a silent miscount.
  LogicalResult refuse(Operation *op, const char *why);

  // The arm of visitOperandDiv where the divisor does not divide the stride:
  // the axis grows a second one, or the state records the period instead.
  LogicalResult divideByReshapingTheAxis(
      PtrState &state, const PtrState &lhsState, int axis, int64_t k, int64_t s,
      int64_t n, Operation *divOp, OpBuilder &builder,
      llvm::function_ref<LogicalResult(const char *)> giveUp);

  // The arm of visitOperandDiv where the offset is not a multiple of the
  // divisor but shares a factor g with it: the axis is cut into rows of g, each
  // row affine, and the row starts become a gather index. Fails, writing
  // nothing, where no such g exists.
  LogicalResult divideIntoRows(PtrState &state, const PtrState &lhsState,
                               int64_t k, Value result, OpBuilder &builder);

  // The deepest named refusal of the pointer being rewritten, or null. THE WALK
  // REACHES THE ADDPTR FIRST and the load that fails is a separate visit, so the
  // name has to be parked on the pointer value to survive the gap.
  const char *lastRefusal = nullptr;
  llvm::DenseMap<Value, const char *> refusalOfPtr;
  void recordNotRewritten(Value ptr);
  void recordRung(Value ptr, Value tptr);
  // `cause` names the arithmetic that could not be a stride; the result says
  // whether the value became a gather/scatter pointer or left entirely.
  // `Chosen` IS NOT A GAP: a handler looked at the op and the index table IS
  // its answer -- a clamp that really fires is not affine, a bit mask that is
  // not a wrap is not one. `Other` is the absence of a handler. Keeping them in
  // one bucket put arith.minsi at the top of a list whose whole use is "what
  // should be built next".
  enum class Demoted { Rem, Div, Sub, Chosen, Other };
  LogicalResult recordDemotion(Demoted cause, LogicalResult rebuilt,
                               Value operand = {});

  // This function is internally used by getLoopIterArgPtrState and
  // getLoopResultPtrState to get the correct PtrState for either an iter-arg or
  // a loop's result.
  //
  // A PtrState of an scf.for's iter-arg is the same as its corresponding
  // init-arg, except that the strides and offsets have to point to the loop's
  // iter-args that were created to carry the offsets and strides.
  //
  // For instance, for a pointer with index i and rank 2, 4 additional args
  // starting at index i + 1 are created. The PtrState's strides and offsets
  // value of the pointer's iter-arg must point to these 4 additionally created
  // iter-args.
  //
  // A similar process is used for getting the PtrState of the loop's i'th
  // result: its strides and offsets have to point to the corresponding stride
  // and offset values returned by the loop.
  PtrState reconcileLoopPtrState(
      scf::ForOp forOp, size_t ptrArgIndex, const PtrState &state,
      llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal);

  DenseSet<Value> maybeStructuredArgs;
  const bool enableMakeGatherScatterTensorPtr;

  // Whether `x // K` splits its axis instead of demoting the access. OFF by
  // default: the tptr it builds has the STATE's rank while the value keeps the
  // TENSOR's, and no backend reads that pair yet.
  const bool splitSteppedAxes;
  // If false, PtrAnalysis will analysis structured ptr while only identify
  // unstructured ptr.
  // If true, PtrAnalysis will caclulate strides and offsets for
  // unstructured pointers. This is used to support gather/scatter access.
  // The default mode is false. Only set to true when caclulating
  // unstructured pointers for gather/scatter access.
  // The reason to have different mode is to support case like:
  //
  // ptr + (row_offsets[:,None] % mod_offset + some_number) +
  //    row_indices[:None]
  //
  // (row_offsets[:,None] % mod_offset + some_number) is structured and
  // has modulo.
  // row_indices[:, None] is unstructured.
  // When visiting the add operation, we need to apply the modulo to
  //   (row_offsets[:,None] % mod_offset + some_number).
  // But we don't have the information about how to apply the modulo.
  //
  // To simplify the analysis, we do the work in two modes:
  // 1. Analyze to identify the unstructured pointers.
  // 2. Analyze to calculate the strides and offsets for unstructured pointers.
  // In mode 1, isAnalysisingUnstructured is set to false, so we only
  //    identify the unstructured pointers and do not calculate the strides and
  //    offsets.
  // When visit the operand again to calculate the offsets and strides for the
  // unstructured state, we'll set isAnalysisingUnstructured to true.
  // This means that we switched to mode 2 now and are analyzing the
  // unstructured pointers and calculating the strides and offsets for them. In
  // mode 2, we know that the pointer is unstructured, so we can just use the
  // value of arith::RemSIOp as offset directly. Once the analysis is done,
  // we'll switch back to mode 1.
  //
  // Note that this is might be a temporary solution, and we may need to
  // revisit this in the future to support more complex cases.
  bool isAnalysisingUnstructured = false;

public:
  PtrAnalysis(bool enableMakeGatherScatterTensorPtr,
              bool splitSteppedAxes = false)
      : enableMakeGatherScatterTensorPtr(enableMakeGatherScatterTensorPtr),
        splitSteppedAxes(splitSteppedAxes) {}
  void initializeMaybeStructuredArgs(Operation *op);

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;

  LogicalResult cutAtMaskDigits(triton::LoadOp op,
                                 const triton::MaskState &mstate, Value &ptr,
                                 SmallVectorImpl<OpFoldResult> &dims,
                                 SmallVector<OpFoldResult> &lows,
                                 OpBuilder &builder);

  IRMapping ptrMap;

  // Recursively parse a Value; call the corresponding
  // function based on the defining operation and argument type.
  LogicalResult visitOperand(Value operand, PtrState &state, const Location loc,
                             OpBuilder &builder);

  // Operand is a result of an scf.for. Such cases occur when there are multiple
  // levels of nested loops where the results of the inner scf.for (pointer) are
  // yielded by the outer loop.
  LogicalResult visitOperandForOp(scf::ForOp forOp, Value operand,
                                  PtrState &state, const Location loc,
                                  OpBuilder &builder);

  // Operand is the result of arith.addi. Process both arguments and insert any
  // arith.addi instruction as needed.
  // Main assumptions:
  //  Only one of lhsState and rhsState has source field set
  //  Current PtrState should be empty
  // Expected result:
  //  source = lhsState.source ? lhsState.source : rhsState.source
  //  sizes[i] = lhsState.sizes[i] (which should match rhsState.sizes[i])
  //  offsets[i] = lhsState.offsets[i] + rhsState.offsets[i]
  //  strides[i] = lhsState.strides[i] + rhsState.strides[i]
  LogicalResult visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // Operand is the result of arith.muli. Process both arguments and insert any
  // arith.muli instruction as needed.
  // Main assumptions:
  //  Neither lhsState nor rhsState has source field set
  //  Current PtrState should be empty
  //  Currently only support one of the operand is a scalar index
  // Expected result (scalar and tensorState represent the two operands):
  //  source = null
  //  sizes[i] = tensorState.sizes[i]
  //  offsets[i] = tensorState.offsets[i] * scalar
  //  strides[i] = tensorState.strides[i] * scalar
  // `(x // K)` IS AFFINE ON TWO AXES, NOT ONE. Reads it into a split rather
  // than refusing; a division does not distribute over a sum, so the guard is
  // on the whole expression and not on one axis of it.
  LogicalResult visitOperandDiv(Operation *divOp, Value lhs, Value rhs,
                                Value result, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // A SUBTRACTION IS AN ADDITION OF A NEGATED SCALAR, and only a scalar:
  // negating a state that varies would put a negative stride on an axis.
  LogicalResult visitOperandSub(arith::SubIOp subOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // The structured reading of a `%`: it wraps an axis. visitOperandRem is the
  // one that decides what to do when this fails. Takes the three values rather
  // than the op because a bit mask spells the same wrap without a RemSIOp.
  LogicalResult visitOperandRemAsWrap(Operation *op, Value lhs, Value rhs,
                                      Value result, PtrState &state,
                                      const Location loc, OpBuilder &builder);

  LogicalResult visitOperandRem(Operation *op, Value lhs, Value rhs,
                                Value result, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // An unsigned `//` or `%` of a non-negative value IS the signed one, so this
  // hands both to the signed arm rather than reading them a second time.
  LogicalResult visitOperandUnsigned(Operation *op, PtrState &state,
                                     const Location loc, OpBuilder &builder);

  // `x & (2^k - 1)` IS `x % 2^k` where x cannot be negative, and only the
  // second spelling is a wrap this analysis reads. The sign is the whole
  // soundness argument: `-3 & 7` is 5 and `-3 rem 8` is -3.
  LogicalResult visitOperandAnd(arith::AndIOp andOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // A CLAMP IS NOT AFFINE, so this answers with one of the three things that
  // are: a number, the operand a clamp that never fires leaves alone, or an
  // index table. `isMin` picks which of minsi/maxsi is being read.
  LogicalResult visitOperandClamp(Operation *op, bool isMin, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  // Operand is the result of make_range.
  // Main assumptions:
  //  start, end, and shape are all statically known
  //  The output of make_range is 1-dimensional
  //  Does not check validity of inputs (e.g., stride > 0)
  // Expected result:
  //  source = null
  //  sizes[0] = shape[0]
  //  offset[0] = start
  //  strides[0] = ceiling( (end - start) / shape[0] )
  LogicalResult visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                      PtrState &state, Location loc,
                                      OpBuilder &builder);

  // Operand is the result of expand_dims
  // Main assumptions:
  //  Only 1 dimension changes for each invocation of reshape
  //  The changed dimension must have size of 1
  // Expected result:
  //  Insert a dimension of size 1, stride 0, and offset 0
  LogicalResult visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder);

  // Operand is the result of broadcast
  // Main assumptions:
  //  Rank of soure and result is the same
  // Expected result:
  //  Update sizes[i] only, no changes to other fields
  LogicalResult visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                      PtrState &state, const Location loc,
                                      OpBuilder &builder);

  // Operand is the result of splat
  // Main assumptions:
  //  Source is a scalar value (i.e., an integer or a pointer, not a tensor)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] = 0
  //  if source is an integer, offset[0] = scalar = source
  LogicalResult visitOperandSplat(triton::SplatOp splatOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  // Operand is the result of arith.constant that is a splat
  // Main assumptions:
  //  Source is a constant op that produces a constant dense tensor where all
  //  elements are the same (i.e.: a constant that is splatted)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] =
  //  splat value if i == 0, otherwise 0
  LogicalResult visitOperandConstSplat(arith::ConstantOp op, PtrState &state,
                                       const Location loc, OpBuilder &builder);

  LogicalResult visitOperandExtSI(arith::ExtSIOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  // Operand is the result of addptr.
  // Main assumptions:
  //  The ptr field should populate the source field
  //  ptr and offset fields should result in same rank
  // Expected result:
  //  The resulting state for ptr and offset wil be added
  LogicalResult visitOperandAddptr(triton::AddPtrOp addptrOp, PtrState &state,
                                   const Location loc, OpBuilder &builder);

  // The visit itself; visitOperandAddptr wraps it with the step retry.
  LogicalResult visitAddptrOnce(triton::AddPtrOp addptrOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // Operand is the result of tts.make_tptr.
  // Main assumptions:
  //  This function is only called when rewriting a loop
  // Expected result:
  //  Directly grab all corresponding fields from tts.make_tptr.
  LogicalResult visitOperandMakeTPtr(tts::MakeTensorPtrOp makeTPtrOp,
                                     PtrState &state, const Location loc,
                                     OpBuilder &builder);

  // Operand is the result of tt.int_to_ptr.
  // Expected result:
  //  Directly grab op result
  LogicalResult visitOperandIntToPtr(triton::IntToPtrOp intToPtrOp,
                                     PtrState &state, const Location loc,
                                     OpBuilder &builder);

  // Operand is the result of tt.bitcast.
  // Expected result:
  //  Directly grab op result
  LogicalResult visitOperandBitcast(triton::BitcastOp bitcastOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder);

  // Get the computed PtrState for the forOp's init-arg at the provided index.
  FailureOr<PtrState> getLoopInitArgPtrState(scf::ForOp forOp, size_t index);

  // Get the computed PtrState for the forOp's iter-arg at the provided index.
  FailureOr<PtrState> getLoopIterArgPtrState(scf::ForOp forOp, size_t index);

  // Get the computed PtrState for the forOp's result at the provided index.
  FailureOr<PtrState> getLoopResultPtrState(scf::ForOp forOp, size_t index);

  // After PtrAnalysis finishes, rewrite the GetStructuredStateOp by creating
  // the correct initialization ops for offsets and strides and passing them to
  // any loop's init-args.
  LogicalResult rewriteGetStructuredStateOp(tts::GetStructuredStateOp op);

  // Parse the state of AddPtrOp, insert any instruction needed to
  // calculate strides and offsets, build PtrState for this operand, and record
  // PtrState for knownPtrs.
  LogicalResult rewriteAddptrOp(triton::AddPtrOp op);

  // A POINTER OP THAT DOES NOT MOVE THE ADDRESS -- splat, broadcast,
  // expand_dims -- adds a stride-0 axis and nothing else. visitOperand models
  // all three, but only rewriteOp puts a result into `ptrMap`, and a load
  // reads `ptrMap`. So one of these standing LAST in a pointer chain left a
  // hole there and the load fell through to the unstructured path.
  LogicalResult rewriteStride0PtrOp(Operation *op);

  // A layout conversion of a pointer moves no address: its result stands for the
  // same pointer -- re-made in the new layout when structured, shared when gathered.
  LogicalResult rewriteConvertLayoutOp(triton::gpu::ConvertLayoutOp op);

  // Parse the state of YieldOp, insert any instruction needed to calculate
  // strides and offsets, build PtrState for this operand, and record PtrState
  // in knownPtrs.
  LogicalResult
  rewriteYieldOp(scf::YieldOp op,
                 llvm::SmallDenseMap<int, PtrState> &knownPtrsFor);

  // Rewrite eligible tt.addptr in loop init args so loop can update the such
  // pointers over iterations. Insert any instruction needed to calculate
  // strides, offsets, and modulos.
  LogicalResult rewriteForOp(scf::ForOp op);

  LogicalResult rewriteLoadOp(triton::LoadOp op, bool useUnsafeMask = false);

  LogicalResult rewriteStoreOp(triton::StoreOp op, bool useUnsafeMask = false);

  // `x % W` where every access reading it is masked to `x < W`. Run before
  // the analysis: the remainder is dropped, not read through, because anyone
  // ELSE reading it sees the op too.
  static void dropMaskedModulo(Operation *scope);

  LogicalResult rewriteOp(Operation *op, bool useUnsafeMask = false);
};

} // namespace tts

} // namespace mlir

#endif
