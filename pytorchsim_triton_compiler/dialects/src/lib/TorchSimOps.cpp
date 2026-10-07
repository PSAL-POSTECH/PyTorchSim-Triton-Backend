//===- TorchSimOps.cpp -----------------------------------------*- C++ -*-===//
//
// WHAT IS LEFT AFTER ODS AND THE DPS INTERFACE: the tile is a shaped thing, a
// yielded tile is the one handed in, and each tail segment is as long as the
// attribute that names its axes says -- which ODS cannot state across the two.
//
//===----------------------------------------------------------------------===//
#include "TorchSim/TorchSimOps.h"
#include "TorchSim/TorchSimDialect.h"
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
