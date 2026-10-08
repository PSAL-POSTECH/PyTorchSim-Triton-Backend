//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "TorchsimExt/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "TorchsimExt/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "TorchsimExt/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "TorchsimExt/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-arith-to-linalg"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONARITHTOLINALG
#include "TorchsimExt/Conversion/TritonArithToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class TritonArithToLinalgPass
    : public mlir::triton::impl::TritonArithToLinalgBase<
          TritonArithToLinalgPass> {
  using TritonArithToLinalgBase<
      TritonArithToLinalgPass>::TritonArithToLinalgBase;

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

  // Add additional I32 arguments to represent:
  // - num_programs, 3 in total, one for each axis of the launch grid
  // - program_id, 3 in total, one for each axis of the launch grid
  static void addProgramInfo(triton::FuncOp func) {
    OpBuilder b(func);

    auto origFuncType = func.getFunctionType();
    auto origInputTypes = origFuncType.getInputs();
    SmallVector<Type> newInputTypes(origInputTypes);
    newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

    auto newFuncType =
        b.getFunctionType(newInputTypes, origFuncType.getResults());

    func.setFunctionType(newFuncType);

    // Add empty attributes for each new argument if needed
    if (func.getAllArgAttrs()) {
      SmallVector<DictionaryAttr> newArgAttrs;
      func.getAllArgAttrs(newArgAttrs);
      newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
      func.setAllArgAttrs(newArgAttrs);
    }

    // Add the corresponding arguments to function body
    for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
      func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
    }
  }

  // upstream's decomposition, with the destination made in the concat's OWN
  // type: upstream builds it from shape and element alone, so the layout
  // encoding of a kernel that arrived with layouts is lost on the way and only
  // a cast at the end puts it back. A dynamic shape is left to upstream.
  struct DecomposeConcatKeepingType : OpRewritePattern<tensor::ConcatOp> {
    DecomposeConcatKeepingType(MLIRContext *ctx)
        : OpRewritePattern<tensor::ConcatOp>(ctx, /*benefit=*/2) {}
    LogicalResult matchAndRewrite(tensor::ConcatOp op,
                                  PatternRewriter &rewriter) const override {
      auto resultType = op.getResultType();
      if (!resultType.hasStaticShape() ||
          llvm::any_of(op.getInputs(), [](Value v) {
            return !cast<RankedTensorType>(v.getType()).hasStaticShape();
          }))
        return failure();
      Location loc = op.getLoc();
      int64_t dim = op.getDim();
      Value dest = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});
      SmallVector<OpFoldResult> offsets(resultType.getRank(), rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(resultType.getRank(), rewriter.getIndexAttr(1));
      int64_t at = 0;
      for (Value input : op.getInputs()) {
        auto inputType = cast<RankedTensorType>(input.getType());
        SmallVector<OpFoldResult> sizes = llvm::to_vector(llvm::map_range(
            inputType.getShape(),
            [&](int64_t d) -> OpFoldResult { return rewriter.getIndexAttr(d); }));
        offsets[dim] = rewriter.getIndexAttr(at);
        dest = tensor::InsertSliceOp::create(rewriter, loc, input, dest, offsets,
                                             sizes, strides);
        at += inputType.getDimSize(dim);
      }
      rewriter.replaceOp(op, dest);
      return success();
    }
  };

  LogicalResult applyTensorConcatDecomposition() {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<DecomposeConcatKeepingType>(context);
    tensor::populateDecomposeTensorConcatPatterns(patterns);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return failure();
    }
    return success();
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                triton::TritonDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    {
      RewritePatternSet patterns(&getContext());
      populateTritonArithToLinalgCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }
    }

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        tts::TritonStructuredDialect>();

    target.addLegalOp<ModuleOp>();

    target.addLegalOp<triton::FuncOp, triton::ReturnOp>();

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          // Lower dense constant to linalg.fill
          if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            if (!isa<RankedTensorType>(constOp.getResult().getType())) {
              return true;
            }

            if (auto denseAttr =
                    dyn_cast<DenseElementsAttr>(constOp.getValue())) {
              if (denseAttr.isSplat() &&
                  isa<FloatType, IntegerType>(denseAttr.getElementType())) {
                return false;
              }
            }
            return true;
          }

          bool operateOnTensors =
              llvm::all_of(op->getOperandTypes(), [](Type type) {
                return isa<RankedTensorType>(type);
              });

          return !operateOnTensors;
        });

    if (pidsToFuncArgs) {
      target.addIllegalOp<triton::GetProgramIdOp, triton::GetNumProgramsOp>();
    }

    if (addptrToLinalg) {
      target.addDynamicallyLegalOp<triton::AddPtrOp>([](triton::AddPtrOp op) {
        return !isa<ShapedType>(op.getResult().getType());
      });
    }

    target.addDynamicallyLegalOp<triton::BitcastOp>(
        [this](triton::BitcastOp op) {
          if (!tensorPtrToLinalg) {
            return triton::isPtrTypeLike(op.getType());
          } else {
            if (triton::isPtrTypeLike(op.getType())) {
              return !isa<ShapedType>(op.getType());
            }
            return false;
          }
        });

    // TODO: Might want to consolidate this flag with addptrToLinalg later.
    if (tensorPtrToLinalg) {
      target.addDynamicallyLegalOp<triton::LoadOp, triton::StoreOp,
                                   triton::IntToPtrOp, triton::PtrToIntOp>(
          [](auto op) {
            return !isa<ShapedType>(op->getOperands()[0].getType());
          });
      populateTritonTensorPtrConversionPatterns(patterns);
    }

    if (!assertToCf) {
      target.addLegalOp<triton::AssertOp>();
    }

    triton::populateTritonArithToLinalgConversionPatterns(
        pidsToFuncArgs, addptrToLinalg, assertToCf, transposeReduceToRank0,
        patterns);

    if (pidsToFuncArgs) {
      for (auto func : getOperation().getOps<triton::FuncOp>()) {
        addProgramInfo(func);
      }
    }

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    if (failed(applyTensorConcatDecomposition())) {
      signalPassFailure();
    }

    // Convert tt.func and tt.return into func's counterparts
    if (ttToFuncFunc) {
      moduleOp.walk([&](triton::FuncOp func) {
        OpBuilder builder(func);

        auto name = func.getName();
        auto type = func.getFunctionType();

        SmallVector<DictionaryAttr> argAttrs, resAttrs;
        func.getAllArgAttrs(argAttrs);
        func.getAllResultAttrs(resAttrs);

        auto funcFunc =
            func::FuncOp::create(builder, func.getLoc(), name, type);
        // Preserve the visibility attribute
        funcFunc.setVisibility(func.getVisibility());
        funcFunc.setAllArgAttrs(argAttrs);
        funcFunc.setAllResultAttrs(resAttrs);

        auto &funcFuncBody = funcFunc.getBody();
        auto &funcBody = func.getBody();

        IRMapping map;
        funcBody.cloneInto(&funcFuncBody, map);

        for (Block &block : funcFuncBody.getBlocks()) {
          auto term = block.getTerminator();
          // Only convert to func.return if the terminator is a tt.return.
          // Otherwise, we will accidentally convert cf.br ops which are also
          // considered terminators.
          if (isa<triton::ReturnOp>(term)) {
            builder.setInsertionPoint(term);
            func::ReturnOp::create(builder, func.getLoc(), term->getOperands());
            term->erase();
          }
        }
        func.erase();
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonArithToLinalgPass(bool tensorPtrToLinalg,
                                      bool transposeReduceToRank0) {
  TritonArithToLinalgOptions options;
  options.tensorPtrToLinalg = tensorPtrToLinalg;
  options.transposeReduceToRank0 = transposeReduceToRank0;
  return std::make_unique<TritonArithToLinalgPass>(options);
}
