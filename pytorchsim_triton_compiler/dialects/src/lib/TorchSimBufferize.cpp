//===- TorchSimBufferize.cpp - our DPS ops through one-shot-bufferize ----===//
//
// UPSTREAM'S SHAPE, MIXED. `bufferizeDestinationStyleOpInterface` is static in
// linalg, so this mirrors it; what differs is ours: no region, and the DRAM and
// tag sides are buffers already, which is the mixed mode that comment allows.
//
//===----------------------------------------------------------------------===//
#include "TorchSim/TorchSimDialect.h"
#include "TorchSim/TorchSimOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

template <typename OpTy>
struct DpsOpInterface
    : public DstBufferizableOpInterfaceExternalModel<DpsOpInterface<OpTy>,
                                                     OpTy> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          const BufferizationState &state) const {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    if (op->getNumResults() == 0 &&
        llvm::none_of(op->getOperandTypes(),
                      [](Type t) { return isa<TensorType>(t); }))
      return success();

    SmallVector<Value> operands;
    operands.reserve(op->getNumOperands());
    for (Value from : op->getOperands()) {
      if (!isa<TensorType>(from.getType())) {
        operands.push_back(from);
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, from, options, state);
      if (failed(buffer))
        return failure();
      operands.push_back(*buffer);
    }

    SmallVector<Value> replacements;
    for (OpResult result : op->getOpResults()) {
      auto dst = cast<DestinationStyleOpInterface>(op);
      OpOperand *init = dst.getDpsInitOperand(result.getResultNumber());
      replacements.push_back(operands[init->getOperandNumber()]);
    }

    rewriter.setInsertionPoint(op);
    OperationState made(op->getLoc(), op->getName(), operands, TypeRange{},
                        op->getAttrs());
    made.propertiesAttr = op->getPropertiesAsAttribute();
    rewriter.insert(Operation::create(made));
    replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }
};

} // namespace

namespace torchsim {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TorchSimDialect *) {
    TransferOp::attachInterface<DpsOpInterface<TransferOp>>(*ctx);
    CopyOp::attachInterface<DpsOpInterface<CopyOp>>(*ctx);
    DotOp::attachInterface<DpsOpInterface<DotOp>>(*ctx);
  });
}

} // namespace torchsim
