// pytorchsim-reduce-tensor-assert: each tt.assert of a tensor<..xi1> becomes the assert of the one i1 that
// every element holds (the sum of its elements, a tt.reduce per axis, equals their count), as the lowering
// stops a failed assert at a scalar one.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL_PYTORCHSIMREDUCETENSORASSERT
#define GEN_PASS_DEF_PYTORCHSIMREDUCETENSORASSERT
#include "PyTorchSimExt/Passes.h.inc"

namespace {

Value buildSumOfAxis0(OpBuilder &b, Location loc, Value src) {
  auto red = ReduceOp::create(b, loc, ValueRange{src}, /*axis=*/0);
  Type i32 = b.getI32Type();
  OpBuilder::InsertionGuard guard(b);
  Block *body = b.createBlock(&red.getCombineOp(), {}, {i32, i32}, {loc, loc});
  Value sum = arith::AddIOp::create(b, loc, body->getArgument(0), body->getArgument(1));
  ReduceReturnOp::create(b, loc, ValueRange{sum});
  return red->getResult(0);
}

void rewriteAssertToScalar(AssertOp assertOp) {
  OpBuilder b(assertOp);
  Location loc = assertOp.getLoc();
  auto ty = cast<RankedTensorType>(assertOp.getCondition().getType());
  Value acc = arith::ExtUIOp::create(b, loc, ty.clone(b.getI32Type()), assertOp.getCondition());
  for (int64_t axis = 0; axis < ty.getRank(); ++axis)
    acc = buildSumOfAxis0(b, loc, acc);
  Value count = arith::ConstantIntOp::create(b, loc, ty.getNumElements(), 32);
  Value all = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, acc, count);
  AssertOp::create(b, loc, all, assertOp.getMessage());
  assertOp->erase();
}

class PyTorchSimReduceTensorAssertPass
    : public impl::PyTorchSimReduceTensorAssertBase<PyTorchSimReduceTensorAssertPass> {
public:
  void runOnOperation() override {
    SmallVector<AssertOp> tensorAsserts;
    getOperation().walk([&](AssertOp op) {
      if (isa<RankedTensorType>(op.getCondition().getType()))
        tensorAsserts.push_back(op);
    });
    for (AssertOp op : tensorAsserts)
      rewriteAssertToScalar(op);
  }
};

}
}
}
