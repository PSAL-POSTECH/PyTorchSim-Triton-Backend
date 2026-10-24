#include "PyTorchSim/PyTorchSimDialect.h"
#include "PyTorchSim/PyTorchSimOps.h"

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

// A vcix reads its operands and writes no operand; a result is a tile it makes, so on buffers it is
// an allocation the instruction writes, its `dst`.
struct VcixOpInterface
    : public BufferizableOpInterface::ExternalModel<VcixOpInterface, ::pytorchsim::VcixOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &, const AnalysisState &) const {
    return true;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &, const AnalysisState &) const {
    return false;
  }
  AliasingValueList getAliasingValues(Operation *, OpOperand &, const AnalysisState &) const {
    return {};
  }
  bool bufferizesToAllocation(Operation *, Value) const { return true; }

  FailureOr<BufferLikeType> getBufferType(Operation *, Value value,
                                          const BufferizationOptions &options,
                                          const BufferizationState &,
                                          SmallVector<Value> &) const {
    auto t = cast<TensorType>(value.getType());
    std::optional<Attribute> space = options.defaultMemorySpaceFn(t);
    if (!space)
      return failure();
    return cast<BufferLikeType>(getMemRefTypeWithStaticIdentityLayout(t, *space));
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          const BufferizationState &state) const {
    auto vcix = cast<::pytorchsim::VcixOp>(op);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value> args;
    for (Value from : vcix.getArgs()) {
      if (!isa<TensorType>(from.getType())) {
        args.push_back(from);
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, from, options, state);
      if (failed(buffer))
        return failure();
      args.push_back(*buffer);
    }
    Value dst;
    if (Value r = vcix.getResult()) {
      FailureOr<BufferLikeType> ty = bufferization::getBufferType(r, options, state);
      if (failed(ty))
        return failure();
      FailureOr<Value> alloc =
          options.createAlloc(rewriter, op->getLoc(), cast<MemRefType>(*ty), ValueRange{});
      if (failed(alloc))
        return failure();
      dst = *alloc;
    }
    rewriter.create<::pytorchsim::VcixOp>(op->getLoc(), TypeRange{}, args, dst,
                               vcix.getOpcodeAttr(), vcix.getImmAttr(), vcix.getFormAttr(),
                               vcix.getLayoutsAttr(), vcix.getOutLayoutAttr());
    SmallVector<Value> replacements;
    if (dst)
      replacements.push_back(dst);
    replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }
};

}

namespace pytorchsim {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, PyTorchSimDialect *) {
    TransferOp::attachInterface<DpsOpInterface<TransferOp>>(*ctx);
    ConvertLayoutOp::attachInterface<DpsOpInterface<ConvertLayoutOp>>(*ctx);
    GatherOp::attachInterface<DpsOpInterface<GatherOp>>(*ctx);
    DotOp::attachInterface<DpsOpInterface<DotOp>>(*ctx);
    VcixOp::attachInterface<VcixOpInterface>(*ctx);
  });
}

}
