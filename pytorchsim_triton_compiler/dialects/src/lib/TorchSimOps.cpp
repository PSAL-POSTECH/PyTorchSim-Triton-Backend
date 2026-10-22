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
  if (auto box = getPieceBoxAttr()) {
    if (getCutAttr())
      return emitOpError("a piece of a cut transfer is not cut again");
    auto shaped = dyn_cast<ShapedType>(sram);
    if (shaped && shaped.hasRank() && box.size() != (size_t)std::max<int64_t>(shaped.getRank(), 1))
      return emitOpError("piece_box has ") << box.size() << " entries for a rank-"
             << shaped.getRank() << " tile";
    for (auto [d, entry] : llvm::enumerate(box.getAsRange<DenseI64ArrayAttr>())) {
      ArrayRef<int64_t> lohi = entry.asArrayRef();
      if (lohi.empty())
        continue;
      int64_t extent = shaped && shaped.hasRank() && shaped.getRank() ? shaped.getDimSize(d) : 1;
      if (lohi.size() != 2 || lohi[0] < 0 || lohi[0] >= lohi[1] ||
          (!ShapedType::isDynamic(extent) && lohi[1] > extent))
        return emitOpError("piece_box entry ") << d << " is not a [lo, hi] inside the axis";
    }
  }
  return success();
}

LogicalResult WaitOp::verify() {
  APInt amount;
  if (matchPattern(getAmount(), m_ConstantInt(&amount)) && !amount.isStrictlyPositive())
    return emitOpError("waits for ") << amount.getSExtValue()
           << " elements; a transfer signalling its semaphore moves at least one";
  for (Value v : getLanding()) {
    auto space = dyn_cast_or_null<IntegerAttr>(cast<BaseMemRefType>(v.getType()).getMemorySpace());
    if (!space || space.getInt() != kScratchpadSpace)
      return emitOpError("lands in ") << v.getType() << ", which is not a scratchpad buffer";
  }
  return success();
}

MutableOperandRange TransferOp::getDpsInitsMutable() {
  if (getDmaKind() == "MVOUT")
    return MutableOperandRange(getOperation(), 0, 0);
  return getSramMutable();
}

using EffectList = SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

static void addOnBuffer(EffectList &effects, MemoryEffects::Effect *effect, OpOperand &o) {
  if (isa<BaseMemRefType>(o.get().getType()))
    effects.emplace_back(effect, &o, SideEffects::DefaultResource::get());
}

/// Signals its semaphore in every form. MVIN reads `dram` and writes a buffer `sram`; MVOUT
/// reads a buffer `sram` and writes `dram`; a buffer `index` is read. A tensor `sram` is a value,
/// not memory, and the scalars are no memory at all.
void TransferOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), Semaphore::get());
  bool load = getDmaKind() == "MVIN";
  MemoryEffects::Effect *read = MemoryEffects::Read::get(), *write = MemoryEffects::Write::get();
  effects.emplace_back(load ? read : write, &getDramMutable(), SideEffects::DefaultResource::get());
  addOnBuffer(effects, load ? write : read, getSramMutable());
  for (OpOperand &o : getIndexMutable())
    addOnBuffer(effects, read, o);
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

/// Occupies the systolic array in every form; on buffers reads `a` and `b` and adds into `init`.
void DotOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), UnitState::get());
  addOnBuffer(effects, MemoryEffects::Read::get(), getAMutable());
  addOnBuffer(effects, MemoryEffects::Read::get(), getBMutable());
  addOnBuffer(effects, MemoryEffects::Read::get(), getInitMutable());
  addOnBuffer(effects, MemoryEffects::Write::get(), getInitMutable());
}

/// Occupies the unit that carries it in every form; on buffers reads `src` and writes part of
/// `dst`, which it also reads since a window or broadcast leaves the rest of `dst` as it was.
void CopyOp::getEffects(EffectList &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), UnitState::get());
  addOnBuffer(effects, MemoryEffects::Read::get(), getSrcMutable());
  addOnBuffer(effects, MemoryEffects::Read::get(), getDstMutable());
  addOnBuffer(effects, MemoryEffects::Write::get(), getDstMutable());
}

/// A bank view and a pack are the buffer they are made from, seen with other strides or elements.
Value BankViewOp::getViewSource() { return getSource(); }
Value PackOp::getViewSource() { return getBank(); }

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
