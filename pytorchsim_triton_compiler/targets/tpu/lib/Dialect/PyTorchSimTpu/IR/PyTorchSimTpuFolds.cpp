#include "Dialect/PyTorchSimTpu/IR/PyTorchSimTpuFolds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

template <typename OpTy> static TypeID idOf() { return TypeID::get<OpTy>(); }
template <typename OpTy> static llvm::StringRef nameOf() { return OpTy::getOperationName(); }

/// The XLU folds f32 by max, sum or min. Its max and min keep a NaN (arith.maximumf, minimumf); maxnumf and
/// minnumf drop one, so the lowering pushes the fold's identity where a lane holds a NaN.
pytorchsim::CrossLaneFolds pytorchsim::tpu::crossLaneFolds() {
  static const CrossLaneFold folds[] = {
      {idOf<arith::MaximumFOp>, nameOf<arith::MaximumFOp>, "max", false},
      {idOf<arith::MaxNumFOp>, nameOf<arith::MaxNumFOp>, "max", true},
      {idOf<arith::AddFOp>, nameOf<arith::AddFOp>, "sum", false},
      {idOf<arith::MinimumFOp>, nameOf<arith::MinimumFOp>, "min", false},
      {idOf<arith::MinNumFOp>, nameOf<arith::MinNumFOp>, "min", true},
  };
  return {folds, [](MLIRContext *ctx) -> Type { return Float32Type::get(ctx); }};
}
