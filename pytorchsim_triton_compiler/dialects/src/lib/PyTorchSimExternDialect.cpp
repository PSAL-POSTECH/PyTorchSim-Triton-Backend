#include "PyTorchSim/PyTorchSimExternDialect.h"
#include "PyTorchSim/PyTorchSimExternOps.h"

using namespace mlir;
using namespace pytorchsim;
using namespace pytorchsim::external;

#include "PyTorchSim/PyTorchSimExternOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "PyTorchSim/PyTorchSimExternOps.cpp.inc"

void PyTorchSimExternDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PyTorchSim/PyTorchSimExternOps.cpp.inc"
      >();
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
