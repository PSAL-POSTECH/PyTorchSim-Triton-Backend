#include "TorchSim/TorchSimOps.h"
#include "TorchSim/TorchSimDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;
using namespace torchsim;

#define GET_OP_CLASSES
#include "TorchSim/TorchSimOps.cpp.inc"

LogicalResult TransferOp::verify() {
  Type sram = getSram().getType();
  if (!isa<TensorType>(sram) && !isa<BaseMemRefType>(sram))
    return emitOpError("scratchpad side must be a tensor or a memref, got ")
           << sram;
  if (getNumResults() == 1 && getTile().getType() != sram)
    return emitOpError("the tile it yields is not the type it was handed");
  size_t masked = getMaskedAxes() ? getMaskedAxes()->size() : 0;
  if (getClamp().size() != 2 * masked)
    return emitOpError("clamps ") << getClamp().size() << " bounds for "
           << masked << " masked axes; each takes a low and a high";
  size_t strided = getDynStrideAxes() ? getDynStrideAxes()->size() : 0;
  if (getDynStrides().size() != strided)
    return emitOpError("carries ") << getDynStrides().size()
           << " runtime strides for " << strided << " dyn_stride_axes";
  if (getDynFill() && !masked)
    return emitOpError("a runtime fill with no masked axis fills nothing");
  if (bool(getIndex()) != bool(getOffsetStride()))
    return emitOpError("an index and the stride it is read at come together");
  if (!getIndex() && (getDynOffsetStride() || getIndirectDim()))
    return emitOpError("states how an index is read but has no index");
  return success();
}

LogicalResult WaitOp::verify() {
  APInt amount;
  if (matchPattern(getAmount(), m_ConstantInt(&amount)) && !amount.isStrictlyPositive())
    return emitOpError("waits for ") << amount.getSExtValue()
           << " elements; a batch it waits for moved at least one";
  return success();
}

MutableOperandRange TransferOp::getDpsInitsMutable() {
  if (getDmaKind() == "MVOUT")
    return MutableOperandRange(getOperation(), 0, 0);
  return getSramMutable();
}

LogicalResult DotOp::verify() {
  auto a = cast<ShapedType>(getA().getType());
  auto b = cast<ShapedType>(getB().getType());
  auto c = cast<ShapedType>(getInit().getType());
  int64_t r = c.getRank();
  if (r < 2 || a.getRank() != r || b.getRank() != r)
    return emitOpError("a, b and init share one rank of two or more");
  if (auto maps = getIndexingMaps()) {
    if (maps->size() != 3)
      return emitOpError("states a map for each of a, b and init");
    return success();
  }
  if (a.getDimSize(r - 1) != b.getDimSize(r - 2) || a.getDimSize(r - 2) != c.getDimSize(r - 2) ||
      b.getDimSize(r - 1) != c.getDimSize(r - 1))
    return emitOpError("a is [..,M,K], b [..,K,N] and init [..,M,N]");
  if (getNumResults() == 1 && getResult().getType() != getInit().getType())
    return emitOpError("the product it yields is not the init's type");
  return success();
}

bool VcixOp::isStateful() { return getArgs().empty() || !(getResult() || getDst()); }

LogicalResult VcixOp::verify() {
  if (getResult() && getDst())
    return emitOpError("yields a tile and writes one: a result is the tensor form, dst the buffer's");
  if (getArgs().empty() && !getResult() && !getDst())
    return emitOpError("names no tile");
  if (getLayouts().size() != getArgs().size())
    return emitOpError("states ") << getLayouts().size() << " layouts for " << getArgs().size()
                                  << " operands";
  if (getResult() && !getOutLayout())
    return emitOpError("yields a tile and states no layout for it");
  Value out = getResult() ? Value(getResult()) : Value(getDst());
  for (Value a : getArgs()) {
    if (!isa<RankedTensorType, BaseMemRefType>(a.getType()))
      return emitOpError("operand ") << a.getType() << " is not a tile";
    if (out && cast<ShapedType>(a.getType()).getShape() != cast<ShapedType>(out.getType()).getShape())
      return emitOpError("operand ") << a.getType() << " is not the result's shape "
                                     << out.getType();
  }
  return success();
}

void VcixOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (isStateful()) {
    effects.emplace_back(MemoryEffects::Read::get(), UnitState::get());
    effects.emplace_back(MemoryEffects::Write::get(), UnitState::get());
  }
  for (OpOperand &o : getArgsMutable())
    if (isa<BaseMemRefType>(o.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &o, SideEffects::DefaultResource::get());
  for (OpOperand &o : getDstMutable())
    effects.emplace_back(MemoryEffects::Write::get(), &o, SideEffects::DefaultResource::get());
}

Speculation::Speculatability VcixOp::getSpeculatability() {
  return isStateful() || getDst() ? Speculation::NotSpeculatable : Speculation::Speculatable;
}
