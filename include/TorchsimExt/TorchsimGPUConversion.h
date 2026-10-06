//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonGPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class TorchsimGPUTypeConverter : public TypeConverter {
public:
  TorchsimGPUTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp,
                         int numCTAs, bool enableSourceRemat,
                         bool torchsimDefaultLayout = false);
  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumCTAs() const { return numCTAs; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
  // The torchsim lane machine's default layout: every thread on one axis, the
  // first of extent above one, where its unconstrained tile rides.
  bool torchsimDefaultLayout;
};

class TorchsimGPUConversionTarget : public ConversionTarget {
public:
  explicit TorchsimGPUConversionTarget(MLIRContext &ctx,
                                     TorchsimGPUTypeConverter &typeConverter);

  // Determine whether the operation is currently legal. I.e. it has layouts
  // assigned to its tensor operands and results.
  static bool isDynamicallyLegal(Operation *op,
                                 const TypeConverter &typeConverter);
};

namespace impl {
LogicalResult convertGatherScatterOp(Operation *op, ValueRange operands,
                                     OpOperand &xOffsetsMutable,
                                     const TypeConverter &typeConverter,
                                     ConversionPatternRewriter &rewriter);
} // namespace impl

// Generic pattern for converting a TMA gather or scatter operation.
template <typename OpT>
struct GatherScatterOpPattern : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return impl::convertGatherScatterOp(op, adaptor.getOperands(),
                                        op.getXOffsetsMutable(),
                                        *this->getTypeConverter(), rewriter);
  }
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
