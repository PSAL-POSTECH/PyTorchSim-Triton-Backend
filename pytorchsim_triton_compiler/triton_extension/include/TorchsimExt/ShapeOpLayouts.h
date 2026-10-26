// The layouts of pytorchsim.join, split and cat as linear-layout algebra: the whole is a part with one more
// register bit (a new last axis of 2, or axis 0 doubled), a part the whole without it. Whether a pair is
// legal on the machine is the layout port's question (LayoutModel.h).
#ifndef TORCHSIM_EXT_SHAPE_OP_LAYOUTS_H
#define TORCHSIM_EXT_SHAPE_OP_LAYOUTS_H

#include "TorchsimExt/LayoutModel.h"
#include "mlir/IR/BuiltinTypes.h"

namespace torchsim {

// The layout of each part of `whole` (with its encoding) for a TS_JOIN, TS_SPLIT or TS_CAT.
mlir::Attribute partLayout(TsOpKind kind, mlir::RankedTensorType whole);

// The layout of the whole made of `part` (with its encoding) for a TS_JOIN, TS_SPLIT or TS_CAT.
mlir::Attribute wholeLayout(TsOpKind kind, mlir::RankedTensorType part);

// The layout of a reduce's operand whose result is `result` (with its encoding): the reduced axis of
// `extent` inserted at `axis` on register bits, so each lane reduces what it holds; null where `extent` is
// no power of two.
mlir::Attribute withReducedAxis(mlir::RankedTensorType result, int axis, int64_t extent);

} // namespace torchsim

#endif // TORCHSIM_EXT_SHAPE_OP_LAYOUTS_H
