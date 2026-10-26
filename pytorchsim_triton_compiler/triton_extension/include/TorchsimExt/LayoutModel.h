// The layout port: the one channel through which the TTGIR passes ask a target's pack what its
// machine decides about layouts. The pack registers a table of C callbacks before the passes run;
// with none registered every question keeps upstream's answer.
#ifndef TORCHSIM_EXT_LAYOUT_MODEL_H
#define TORCHSIM_EXT_LAYOUT_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TS_LAYOUT_MODEL_VERSION 3

// One layout as linear-layout bases: the element coordinate each register bit and each lane bit
// moves to, over a tile of `shape`.
typedef struct TsLayout {
  int32_t rank;
  const int64_t *shape;       // rank
  int32_t elem_bits;
  int32_t n_reg;
  int32_t n_lane;
  const int32_t *reg_bases;   // n_reg x rank
  const int32_t *lane_bases;  // n_lane x rank
} TsLayout;

typedef enum TsOpKind {
  TS_LOAD, TS_ARITH, TS_MATH, TS_REDUCE, TS_SCAN,
  TS_JOIN, TS_SPLIT, TS_CAT, TS_RESHAPE, TS_TRANS, TS_GATHER, TS_ADDRESS,
} TsOpKind;

// The questions; a null entry, or a negative answer, keeps upstream's.
typedef struct TsLayoutModel {
  int32_t version;
  int32_t lanes;
  int32_t (*layout_free)(int32_t op);
  int32_t (*layout_legal)(int32_t op, const TsLayout *operand, const TsLayout *result);
  double (*convert_cost)(const TsLayout *src, const TsLayout *dst);
  double (*remat_cost)(int32_t op, const TsLayout *at);
  int32_t (*lane_folds)(const char *combiner, int32_t elem_bits);
} TsLayoutModel;

void torchsim_set_layout_model(const TsLayoutModel *model);

#ifdef __cplusplus
}

#include "mlir/IR/BuiltinTypes.h"
#include <optional>

namespace torchsim {

// The registered model, or null when the pack registered none (or another version).
const TsLayoutModel *layoutModel();

// Whether the model says `op` lowers in any layout; false when it gives no answer.
bool layoutFree(TsOpKind op);

// Whether `op` may take `operand` into `result` (each a type with its layout); true when the model
// gives no answer.
bool layoutLegal(TsOpKind op, mlir::RankedTensorType operand, mlir::RankedTensorType result);

// The model's cycles for converting `src` into `dstEnc`, or none when it gives no answer.
std::optional<double> convertCost(mlir::RankedTensorType src, mlir::Attribute dstEnc);

// Whether the target folds a reduce combining by `combiner` (an op name) across lanes; true when the
// model gives no answer.
bool laneFolds(llvm::StringRef combiner, int32_t elemBits);

// The model's cycles for moving each element of `from` to its row-major place in `to` (a reshape's
// operand and result, each with its layout), or none when it gives no answer.
std::optional<double> moveCost(mlir::RankedTensorType from, mlir::RankedTensorType to);

// The model's cycles for making an op of `kind` in the layout of `at`, or none when it gives no answer.
std::optional<double> rematCost(TsOpKind kind, mlir::RankedTensorType at);

} // namespace torchsim
#endif

#endif // TORCHSIM_EXT_LAYOUT_MODEL_H
