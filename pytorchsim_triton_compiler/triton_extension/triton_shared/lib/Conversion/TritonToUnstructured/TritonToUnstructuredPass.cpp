//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
//
////////////////////////////////////////////////////////////////////////////////
// Overview
////////////////////////////////////////////////////////////////////////////////
//
// This pass attempts to lower all loads and stores of unstructured pointers to
// tts.gather or tts.scatter that take a single base, a tensor of offsets, an
// optional tensor of mask values, and a default value in case of load.
//
// In addition, all pointer-producing ops will be eliminated and replaced by
// offset-producing ops. tts.gather and tts.scatter will use the pointer
// directly from the kernel arguments as opposed to pointer produced by ops such
// as tt.addptr and tt.splat.
//
// Example:
//
// %12 = tts.gather %arg0[%10] : (<f32>, tensor<64xi64>) -> tensor<64xf32>
// tts.scatter %12 into %arg1[%arg3] : tensor<64xf32> into (<f32>,
// tensor<64xi32>)
//
// Current assumptions and limitations:
//   - For simplicity, the pass assumes that gather / scatter operations load /
//   store from / to a single base with a tensor of random offsets. As a
//   result, the following triton program would not work:
//
//    @triton.jit
//    def gather_simple(in0, in1, out0):
//        offs = tl.arange(0, 8)
//        in0_ptrs = in0 + offs
//        in1_ptrs = in1 + offs
//        ptrs = tl.cat(in0_ptrs, in1_ptrs, can_reorder=True)
//        c = tl.load(ptrs)
//        out_offs = tl.arange(0, 16)
//        tl.store(out0 + out_offs, c)
//
//   In the above program, `ptrs` contains 2 bases: `in0` and `in1` after the
//   `cat` operation.
//
////////////////////////////////////////////////////////////////////////////////
// Future work
////////////////////////////////////////////////////////////////////////////////
//
// Future work may include scaling the algorithm to support such cases -- one
// possible solution is to let tts.gather and tts.scatter take in an additional
// tensor of base pointers corresponding to the tensor of offsets. But because
// we do not want pointer-producing ops to be present after this pass, we can
// use a tensor of index where each element indicates the index of the pointer
// argument to be used. The drawback is a gather or scatter operation now needs
// one extract lookup to get the base which will affect performance.
//
////////////////////////////////////////////////////////////////////////////////
// Algorithm
////////////////////////////////////////////////////////////////////////////////
//
// Because the goal of triton-shared is to eventually lower all triton ops and
// types to mlir, we want to transform the IR such that the usages of triton
// pointers are as limited as possible. Doing so will help simplify conversion
// to mlir dialects in subsequent passes. In a familiar fashion to the
// triton-to-structured pass, we want triton pointers to only appear in
// tts.gather and tts.scatter only.
//
// With that goal in mind, we want to revisit the triton pointer type.
//
// Triton pointers are created and manipulated through a sequence of ops such as
// tt.addptr, tt.splat, or tt.broadcast. If a triton pointer is created
// through `tt.addptr %ptr %offset`, the new pointer will contain the same base
// pointer as the original pointer; its offset will also be accumulated.
//
// Triton pointers created through tt.splat and tt.broadcast retain their base
// pointers and offsets. Tensors of pointers, however, may have different bases
// when tl.cat is present. For simplicity, we assume tl.cat isn't present as
// mentioned in the overview section.
//
// Therefore, a single triton pointer (tt.ptr) has two pieces of info that is
// implicit:
//   - a base pointer which comes from the kernel arguments
//   - an offset which could be either a tensor of offset or a single integer
//   offset
//
// Leveraging this insight, in order to limit the usages of triton pointer, we
// can explicitly compute and split the above two pieces of info. So chains of
// tt.addptr, tt.splat, and tt.broadcast which produce triton pointers can be
// transformed to sequences of offset (of integer type) manipulation ops and a
// base pointer which comes from the kernel arguments. With this approach, only
// tts.gather and tts.scatter need to be aware of the pointer type.
//
// In essence, this pass transforms all sequences of tt.addptr into sequences of
// offset accumulation ops which are then fed into a single op
// tts.gather or tts.scatter that takes:
//
//   - a base pointer from the kernel arguments
//   - a tensor of offsets (or single offset) that indicates the offsets from
//   the base pointer
//
// All intermediate tt.addptr ops are converted to arith.addi ops that compute
// the offsets. Offsets start at 0 with the provided bit-width. All pointer
// shape manipulation ops such as tt.splat and tt.broadcast will instead operate
// on the offsets and will be converted to linalg in triton-arith-to-linalg.
//
// By default, the pass uses i32 for the initial offsets of all pointers
// (configurable via offset-bit-width=width). If any intermediate tt.addptr
// introduces a larger bitwidth offset, the offsets will be sign-extended to the
// larger bitwidth.
//
////////////////////////////////////////////////////////////////////////////////
// Algorithm
////////////////////////////////////////////////////////////////////////////////
//
// This pass uses a standard worklist-based algorithm to walk the use-def chains
// of all pointer arguments and create replacement ops that operate on offsets
// instead of tt.ptr types.
//
// In cases such as tt.addptr, tt.splat, and tt.broadcast, we create
// corresponding replacement ops which will then be used to map the results
// at the end of the algorithm. We do not want to modify these ops in-place
// because the use-def chains may be changed. In special cases like scf.for, we
// also set the type of the iter-arg and result directly which is usually frown
// upon (but justified).
//
// This approach is used in favor of the traditional ConversionPatternRewriter
// which converts all pointer type into an offset integer type because
// TypeConverter does not support dynamic type based on value. This limitation
// means we have to decide the same bitwidth for all tt.addptr sequences which
// is not ideal.
//
// For instance, assuming we have two sequences of tt.addptr: one operates on
// 32-bit offsets while the other operates on 64-bit offsets. If we set the
// default bitwidth to 64, the 32-bit sequence will require unncessary
// sign-extending when computing the offsets. Contrast this with the manual
// approach, we will only sign-extend where necessary.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <queue>

#define DEBUG_TYPE "triton-to-unstructured"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TRITONTOUNSTRUCTURED
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h.inc"
#undef GEN_PASS_DEF_TRITONTOUNSTRUCTURED

namespace {

// Given a type, return the offset type corresponding to that type with the
// specified width.
// If the type is a tensor, return a tensor of offsets of the same shape. If the
// type is a pointer, return a single offset type.
static Type getPtrOffsetType(Type type, unsigned int bitWidth) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto ptrType =
            dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      // The offset tensor keeps the pointer tensor's layout: the encoding is a
      // fact about where its elements sit, and the addresses sit the same way.
      return RankedTensorType::get(
          tensorType.getShape(), IntegerType::get(type.getContext(), bitWidth),
          tensorType.getEncoding());
    }
  }

  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    return IntegerType::get(type.getContext(), bitWidth);
  }

  llvm_unreachable("unexpected type");
  return nullptr;
}

static unsigned int getBitWidth(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto integerType = dyn_cast<IntegerType>(tensorType.getElementType())) {
      return integerType.getWidth();
    }
  } else if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.getWidth();
  }

  llvm_unreachable("unexpected type");
  return 0;
}

// EVERY ARM THAT CANNOT CARRY AN OFFSET ENDS HERE, and they all end the same
// way: processUnstructuredPtrs returns, the module keeps its `tt.load` and
// `tt.store`, and the lowering below takes the chain.
//
// A WARNING AND NOT AN ERROR, because that is what the pass then DOES. Three
// arms wrote `emitError` and exited 0 -- a program that compiles and answers
// correctly does not get to print "error" -- and two more said nothing at all,
// so which report a reader got depended on which spelling they wrote.
static LogicalResult declineChain(Operation *op, StringRef why) {
  op->emitWarning() << "this pass cannot carry a pointer offset through: "
                    << why
                    << "; the module keeps its tt.load / tt.store for the "
                       "lowering below";
  return failure();
}

class TritonToUnstructuredPass
    : public ::impl::TritonToUnstructuredBase<TritonToUnstructuredPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  struct PtrOffset {
    // the source pointer which comes from the kernel argument
    Value ptr;
    // the pointer type that corresponds to this offset; used when
    // creating tts.make_unstructured_tptr
    Type ptrType;
    // bitwidth that is used for this offset, used to track if sign-extension is
    // necessary
    unsigned int bitWidth;
    // the offset value
    Value offset;
  };

  LogicalResult processUnstructuredPtrs(unsigned int defaultBitWidth = 32) {
    llvm::SmallDenseSet<Value> ptrArgs;
    llvm::DenseMap<Value, PtrOffset> offsetMap;
    std::queue<Value> workList;

    getOperation().walk([&](FunctionOpInterface func) {
      for (auto arg : func.getArguments()) {
        if (!triton::isPtrTypeLike(arg.getType())) {
          continue;
        }

        OpBuilder b(func->getRegion(0));
        Value zero = arith::ConstantOp::create(
            b, arg.getLoc(),
            b.getIntegerAttr(IntegerType::get(&getContext(), defaultBitWidth),
                             0));

        ptrArgs.insert(arg);
        offsetMap.insert({arg, {arg, arg.getType(), defaultBitWidth, zero}});
        workList.push(arg);
      }
    });

    getOperation().walk([&](triton::IntToPtrOp op) {
      // We only want to handle single source pointer,
      // skip if this op produces tensor of pointers
      if (isa<RankedTensorType>(op.getType())) {
        return;
      }
      auto res = op.getResult();
      OpBuilder b(op);
      Value zero = arith::ConstantOp::create(
          b, op.getLoc(),
          b.getIntegerAttr(IntegerType::get(&getContext(), defaultBitWidth),
                           0));

      offsetMap.insert({res, {res, res.getType(), defaultBitWidth, zero}});
      workList.push(res);
    });

    llvm::SmallVector<Operation *> toDelete;

    // WHICH OPS CONSUME A TRACKED POINTER DIRECTLY, said once. One list per
    // kind, because the rewrite below does different work for each: a shared
    // list would need that claim spelled a second time to take it apart again.
    llvm::SmallVector<triton::LoadOp> loadUsers;
    llvm::SmallVector<triton::StoreOp> storeUsers;
    llvm::SmallVector<tts::MakeTensorPtrOp> structuredPtrUsers;

    while (!workList.empty()) {
      auto val = workList.front();
      workList.pop();

      for (auto &use : val.getUses()) {
        auto user = use.getOwner();

        auto res =
            llvm::TypeSwitch<Operation *, LogicalResult>(user)

                .Case<triton::PtrToIntOp>([&](triton::PtrToIntOp op) {
                  if (isa<RankedTensorType>(op.getType())) {
                    return declineChain(op, "a tt.ptr_to_int on a TENSOR of "
                                            "pointers has no single address to "
                                            "materialise the offset onto");
                  }

                  auto offsetInfo = offsetMap.at(op.getSrc());

                  OpBuilder b{op};
                  // We are converting a pointer to an integer here,
                  // materialized the pointer using the accumulated offset
                  // that we have stored so far.
                  auto materializedAddPtr = triton::AddPtrOp::create(
                      b, op->getLoc(), offsetInfo.ptrType, offsetInfo.ptr,
                      offsetInfo.offset);

                  // Change the op to use the "simplified" pointer above.
                  // This should not affect the traversal of uses, but hacky.
                  // We will need to revisit how we process the IRs in this pass
                  // later.
                  op->setOperand(0, materializedAddPtr);

                  return success();
                })
                .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptr) {
                  if (addptr->getParentOfType<scf::IfOp>()) {
                    return declineChain(
                        addptr, "an addptr inside an scf.if may take its base "
                                "from either branch, and an offset names one");
                  }

                  OpBuilder b{addptr};
                  auto loc = addptr->getLoc();

                  auto offsetInfo = offsetMap.at(addptr.getPtr());

                  auto prevOff = offsetInfo.offset;
                  auto off = addptr.getOffset();

                  auto lhsWidth = offsetInfo.bitWidth;
                  auto rhsWidth = getBitWidth(off.getType());
                  auto resWidth = std::max(lhsWidth, rhsWidth);

                  // A widened offset keeps the layout of the offset it
                  // widens, which the recorded pointer type need not carry.
                  auto widened = [&](Value v) -> Type {
                    auto w = IntegerType::get(v.getContext(), resWidth);
                    if (auto t = dyn_cast<RankedTensorType>(v.getType()))
                      return t.clone(w);
                    return w;
                  };
                  if (lhsWidth < resWidth) {
                    prevOff = arith::ExtSIOp::create(b, loc, widened(prevOff),
                                                     prevOff);
                  }

                  if (rhsWidth < resWidth) {
                    off = arith::ExtSIOp::create(b, loc, widened(off), off);
                  }

                  auto accumulatedOff = arith::AddIOp::create(
                      b, loc, getPtrOffsetType(addptr.getType(), resWidth),
                      prevOff, off);

                  PtrOffset newOffsetInfo{offsetInfo.ptr, addptr.getType(),
                                          resWidth, accumulatedOff};

                  offsetMap.insert({addptr, newOffsetInfo});
                  workList.push(addptr);
                  toDelete.push_back(addptr);

                  return success();
                })
                .Case<triton::SplatOp, triton::BroadcastOp,
                      triton::ExpandDimsOp>([&](Operation *op) {
                  auto res = op->getResult(0);
                  auto resType = res.getType();

                  if (!triton::isPtrTypeLike(resType)) {
                    return success();
                  }

                  auto ptr = op->getOperand(0);
                  auto offsetInfo = offsetMap.at(ptr);

                  // THE ATTRIBUTES COME TOO. Cloning by name with operands
                  // and result types alone drops them, and tt.expand_dims
                  // REQUIRES `axis`: the three ops share this arm, and two of
                  // them survived only because they require none.
                  OpBuilder b{op};
                  auto clone =
                      b.create(op->getLoc(), op->getName().getIdentifier(),
                               ValueRange{offsetInfo.offset},
                               TypeRange{getPtrOffsetType(
                                   resType, offsetInfo.bitWidth)},
                               op->getAttrs());

                  PtrOffset newOffsetInfo{offsetInfo.ptr, resType,
                                          offsetInfo.bitWidth,
                                          clone->getResult(0)};

                  offsetMap.insert({
                      res,
                      newOffsetInfo,
                  });
                  workList.push(res);
                  toDelete.push_back(op);

                  return success();
                })
                .Case<triton::BitcastOp>([&](triton::BitcastOp op) {
                  // A BITCAST OF A POINTER CHANGES THE TYPE AND NOT THE
                  // ADDRESS, so the offset accumulated so far passes through
                  // unchanged and only the pointer type it is remembered under
                  // moves. That is the same statement the reshape-like cases
                  // above make, and the same test: a bitcast of a VALUE is not
                  // this pass's business.
                  //
                  // TRITON PUTS ONE IN THE MIDDLE OF EVERY BOOL LOAD. A
                  // `torch.bool` buffer arrives as `!tt.ptr<i1>` and is read
                  // through `!tt.ptr<i8>`, so any kernel with a boolean mask
                  // has a bitcast in its pointer chain -- an attention causal
                  // mask, for one.
                  //
                  // WITHOUT THIS THE POINTER TYPE IS LOST AND THE FAILURE MOVES
                  // rather than showing here: the walk reports "unexpected op
                  // in ptr sequence" and carries on, the gather below is then
                  // built on the `!tt.ptr<i1>` base while triton's `other` for
                  // that load is the i8 it actually reads, and the op fails its
                  // own verifier -- "'tts.gather' op failed to verify that
                  // other matches ptr type", three passes from the cause.
                  auto res = op.getResult();
                  if (!triton::isPtrTypeLike(res.getType())) {
                    return success();
                  }

                  // NOT ONE PtrAnalysis HAS ALREADY TAKEN. It stops recursing
                  // at a bitcast whose result is not shaped
                  // (`visitOperandBitcast`) and makes the bitcast itself the
                  // SOURCE of a `tts.make_tptr` -- that pointer is structured
                  // and finished, and entering it here drags the make_tptr
                  // into a rewrite it was never part of. Measured: a segfault
                  // reading a dead value's type in the MakeTensorPtrOp case
                  // below (kernels/coverage/ptr/ptr_bitcast_scalar_base.py).
                  //
                  // ASKED OF THE USES, not of the type. Both spellings are
                  // scalar `!tt.ptr<T>` -- Mistral's bool mask collapses to a
                  // single element and its bitcast feeds a raw `tt.load` that
                  // this walk MUST convert -- so the shape cannot tell them
                  // apart and what they do with the pointer can.
                  //
                  // AND ASKED AS A DIALECT, not as a list of op names. This
                  // named `tts.make_tptr` alone and PtrAnalysis takes a pointer
                  // onto TWO rungs -- a narrowing bitcast base landed under
                  // `tts.make_gather_scatter_tptr` and was walked into anyway,
                  // reporting "would rescale every offset" about a pointer that
                  // was already finished. Nothing else puts a tts op here: the
                  // gathers and scatters this walk builds come after it.
                  if (llvm::any_of(res.getUsers(), [](Operation *user) {
                        auto *d = user->getDialect();
                        return d && d->getNamespace() == "tts";
                      })) {
                    return success();
                  }

                  // ONLY IF THE SOURCE IS ONE THIS WALK IS TRACKING. A scalar
                  // pointer bitcast can also be the BASE that PtrAnalysis
                  // already chose -- `visitOperandBitcast` stops recursing when
                  // the result is not shaped and makes the bitcast itself the
                  // source of a `tts.make_tptr` -- and that one is nobody's
                  // use here, so its operand was never entered in the map.
                  // Asking `at()` for it is a segfault three passes from
                  // anything that could explain it (measured:
                  // kernels/coverage/ptr/ptr_bitcast_scalar_base.py, stage 3
                  // exit -11).
                  auto it = offsetMap.find(op.getSrc());
                  if (it == offsetMap.end()) {
                    return success();
                  }
                  auto offsetInfo = it->second;

                  // ASKED ONLY ONCE THIS WALK OWNS THE POINTER. The offset in
                  // `offsetInfo` is counted in ELEMENTS, so a cast that changes
                  // the element size rescales it -- but asking before the two
                  // checks above reported that about pointers PtrAnalysis had
                  // already finished, which is how a passing kernel came to
                  // print an error. One predicate, asked where it applies.
                  // ASKED ONLY ONCE THIS WALK OWNS THE POINTER, and answered
                  // the way visitOperandBitcast answers it: the offset is
                  // counted in ELEMENTS, so crossing the cast multiplies it.
                  // Asking before the checks above reported this about pointers
                  // PtrAnalysis had already finished.
                  unsigned ratio = triton::ptrCastElementRatio(op);
                  if (ratio == 0) {
                    return declineChain(op,
                                        "a pointer bitcast that WIDENS an "
                                        "element would divide every offset");
                  }

                  // THE BASE MOVES TOO, and that is the whole of the fix. The
                  // offset is in ELEMENTS and `ptr` is the kernel argument the
                  // gather is finally built on, so a cast that changes the
                  // pointee has to reach both -- `tts.gather`'s verifier says
                  // exactly this ("other matches ptr type": `other` must be
                  // getPointeeType(ptr)), and triton hands down an i8 `other`
                  // for a bool load because i8 is what it reads. Carrying only
                  // the tensor type here leaves the gather on the i1 argument
                  // with an i8 fill, and the op fails to verify.
                  // The new base takes the RESULT's own pointer type rather
                  // than a rebuilt one, so the address space and everything
                  // else come along unchanged. `isPtrTypeLike` is true of a
                  // plain `!tt.ptr<T>` as well as a tensor of them, and both
                  // reach here -- a scalar cast IS the base already.
                  OpBuilder bb{op};
                  auto resTy = res.getType();
                  auto elemPtrTy =
                      isa<RankedTensorType>(resTy)
                          ? cast<RankedTensorType>(resTy).getElementType()
                          : resTy;
                  auto newBase = triton::BitcastOp::create(
                      bb, op->getLoc(), elemPtrTy, offsetInfo.ptr);

                  Value newOffset = offsetInfo.offset;
                  if (ratio > 1) {
                    auto oty = newOffset.getType();
                    Value k;
                    if (auto tty = dyn_cast<RankedTensorType>(oty)) {
                      k = arith::ConstantOp::create(
                          bb, op->getLoc(),
                          DenseElementsAttr::get(
                              tty, APInt(tty.getElementType()
                                             .getIntOrFloatBitWidth(),
                                         ratio)));
                    } else {
                      k = arith::ConstantOp::create(
                          bb, op->getLoc(), bb.getIntegerAttr(oty, ratio));
                    }
                    newOffset =
                        arith::MulIOp::create(bb, op->getLoc(), newOffset, k);
                  }

                  PtrOffset newOffsetInfo{newBase, res.getType(),
                                          offsetInfo.bitWidth, newOffset};
                  offsetMap.insert({res, newOffsetInfo});
                  workList.push(res);
                  return success();
                })
                .Case<tts::MakeGatherScatterTensorPtrOp>(
                    [&](Operation *op) { return success(); })
                // THE ONLY PLACE THAT SAYS WHICH OPS CONSUME A TRACKED
                // POINTER. Each arm collects into a list of its own type and
                // the rewrite below walks those lists, so there is no second
                // spelling of this set to drift from -- an op reaches the
                // rewrite only through the arm that put it in a list.
                .Case<triton::LoadOp>([&](triton::LoadOp load) {
                  loadUsers.push_back(load);
                  return success();
                })
                .Case<triton::StoreOp>([&](triton::StoreOp store) {
                  storeUsers.push_back(store);
                  return success();
                })
                .Case<tts::MakeTensorPtrOp>([&](tts::MakeTensorPtrOp
                                                    makeTensorPtr) {
                  // Special case:
                  // We do not want to create "unstructured tensor pointer" into
                  // tts.make_tptr if the base pointer is directly from the
                  // kernel arguments.
                  if (ptrArgs.contains(makeTensorPtr.getBase())) {
                    return success();
                  }

                  structuredPtrUsers.push_back(makeTensorPtr);
                  return success();
                })
                .Case<scf::ForOp>([&](scf::ForOp forOp) {
                  // Index of the init-arg corresponding to this use, note that
                  // we have to subtract by 3 from the operand number because
                  // scf.for ops always have 3 leading operands for start, end,
                  // and step.
                  auto argIndex = use.getOperandNumber() - 3;
                  auto init = forOp.getInitArgs()[argIndex];

                  auto offsetInfo = offsetMap.at(init);

                  auto offsetType =
                      getPtrOffsetType(offsetInfo.ptrType, offsetInfo.bitWidth);

                  // We're setting both the types of the iter-arg and the
                  // corresponding result directly to the offset type.
                  // At this point, the IR is in an invalid state because the
                  // init-args still have tt.ptr. But at the end, we will
                  // replace all uses of the tt.ptr to offset values.
                  auto iterArg = forOp.getRegionIterArg(argIndex);
                  iterArg.setType(offsetType);

                  auto res = forOp.getResult(argIndex);
                  res.setType(offsetType);

                  // For other ops, we only need to push the result into the
                  // worklist. But for scf.for, the iter-arg corresponding to
                  // the init-arg is used in the op's body instead, we have to
                  // process uses of the iter-arg.
                  PtrOffset iterArgOffset{offsetInfo.ptr, offsetInfo.ptrType,
                                          offsetInfo.bitWidth, iterArg};
                  offsetMap.insert({
                      iterArg,
                      iterArgOffset,
                  });

                  PtrOffset resOffset{offsetInfo.ptr, offsetInfo.ptrType,
                                      offsetInfo.bitWidth, res};
                  offsetMap.insert({
                      res,
                      resOffset,
                  });
                  workList.push(iterArg);
                  workList.push(res);

                  return success();
                })
                .Case<scf::YieldOp>([](auto) { return success(); })
                .Case<triton::gpu::ConvertLayoutOp>([&](triton::gpu::ConvertLayoutOp op) {
                  // A LAYOUT CONVERSION MOVES NO ADDRESS: the offset converts
                  // the same way and the arm reads like splat's.
                  auto res = op.getResult();
                  if (!triton::isPtrTypeLike(res.getType())) {
                    return success();
                  }
                  auto offsetInfo = offsetMap.at(op.getSrc());
                  OpBuilder b{op};
                  auto clone = b.create(
                      op->getLoc(), op->getName().getIdentifier(),
                      ValueRange{offsetInfo.offset},
                      TypeRange{getPtrOffsetType(res.getType(), offsetInfo.bitWidth)},
                      op->getAttrs());
                  PtrOffset newOffsetInfo{offsetInfo.ptr, res.getType(),
                                          offsetInfo.bitWidth, clone->getResult(0)};
                  offsetMap.insert({res, newOffsetInfo});
                  workList.push(res);
                  toDelete.push_back(op);
                  return success();
                })
                .Case<triton::CatOp>([](triton::CatOp op) {
                  return declineChain(
                      op, "tt.cat joins two bases, and an offset names one");
                })
                .Default([&](Operation *op) {
                  // WHERE A SPELLING NOBODY LISTED LANDS. The arms above are
                  // this pass's whole answer to "does this op move the
                  // address"; anything else that uses a tracked pointer is a
                  // spelling neither this list nor PtrAnalysis::visitOperand
                  // was taught, and it arrives here rather than silently.
                  return declineChain(op, "no arm carries an offset through "
                                          "this op, so the spelling is one "
                                          "neither this pass nor "
                                          "PtrAnalysis::visitOperand knows");
                });

        if (failed(res)) {
          return failure();
        }
      }
    }

    // THE REWRITE, ONE LOOP PER KIND. The collector's arms are the only
    // statement of which ops land here, so a kind with no loop is a list
    // nothing reads -- not an op that reaches a Default at run time.
    for (auto load : loadUsers) {
      OpBuilder b{load};
      auto loc = load->getLoc();
      auto offsetInfo = offsetMap.at(load.getPtr());

      auto other = load.getOther();

      if (other) {
        other = tts::utils::getScalarValue(other, loc, b);
        if (!other) {
          return declineChain(load, "the masked load's `other` is not one "
                                    "value every lane shares");
        }
      }

      auto gather = tts::GatherOp::create(
          b, loc, load.getType(), offsetInfo.ptr, offsetInfo.offset,
          load.getMask(), other);

      load->replaceAllUsesWith(gather->getResults());
      load->erase();
    }

    for (auto store : storeUsers) {
      OpBuilder b{store};
      auto loc = store->getLoc();
      auto offsetInfo = offsetMap.at(store.getPtr());
      tts::ScatterOp::create(b, loc, offsetInfo.ptr,
                             offsetInfo.offset, store.getValue(),
                             store.getMask());
      store->erase();
    }

    auto rewriteMakeTensorPtr = [&](tts::MakeTensorPtrOp makeTensorPtr)
        -> LogicalResult {
      OpBuilder b{makeTensorPtr};
      auto loc = makeTensorPtr->getLoc();
      auto offsetInfo = offsetMap.at(makeTensorPtr.getBase());
      auto baseOffset = offsetInfo.offset;

      makeTensorPtr.getBaseMutable().set(offsetInfo.ptr);

      auto &offsetOpnd = makeTensorPtr.getOffsetsMutable()[0];
      auto currOffset = offsetOpnd.get();

      auto baseOffType = baseOffset.getType();
      auto currOffType = currOffset.getType();

      if (baseOffType != currOffType) {
        if (!baseOffType.isIndex() && !currOffType.isIndex() &&
            baseOffType.getIntOrFloatBitWidth() >
                currOffType.getIntOrFloatBitWidth()) {
          makeTensorPtr.emitWarning(
              "truncating offsets which may result in data loss");
        }
        baseOffset = getValueOrCreateCastToIndexLike(
            b, loc, currOffType, baseOffset);
      }

      auto accumulatedOffset = arith::AddIOp::create(
          b, loc, currOffset.getType(), baseOffset, currOffset);

      offsetOpnd.set(accumulatedOffset);

      return success();
    };

    for (auto makeTensorPtr : structuredPtrUsers)
      if (failed(rewriteMakeTensorPtr(makeTensorPtr)))
        return failure();

    for (auto op : toDelete) {
      auto ptrInfo = offsetMap.at(op->getResult(0));
      op->replaceAllUsesWith(ValueRange{ptrInfo.offset});
      op->erase();
    }

    return success();
  }

  void runOnOperation() override {
    if (failed(processUnstructuredPtrs(offsetBitWidth))) {
      getOperation()->emitWarning(
          "Cannot transform tensor of pointers into a single base pointer "
          "with tensor of offsets");
      return;
    }

    PassManager pm(&getContext(), getOperation().getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToUnstructuredPass() {
  return std::make_unique<TritonToUnstructuredPass>();
}
