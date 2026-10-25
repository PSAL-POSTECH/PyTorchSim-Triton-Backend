// The layout questions only a machine answers, for tritongpu-torchsim-anchor-layouts: which axis
// its matrix unit takes on the lanes, and which reduces its lanes fold. The anchor asks them and
// writes the layouts the answers fix; every other anchor rule is the bank memory model's own.
#pragma once
#include "llvm/ADT/DenseMap.h"
#include <string>
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu {

// How a dot meets the matrix unit: every operand with `axis` on the lanes, and when `turned`,
// computed as C^T = B^T A^T -- B the unit's A, A its B.
struct DotForm {
  unsigned axis = 0;
  bool turned = false;
};

class LayoutRules {
public:
  virtual ~LayoutRules() = default;
  // Every dot's form at once: one dot's form can hang on the layouts another dot's values meet.
  virtual llvm::DenseMap<Operation *, DotForm> dotForms(ArrayRef<DotOp> dots,
                                                        unsigned lanes) const = 0;
  // Whether the lanes fold this reduce across themselves, so its axis may stay on them.
  virtual bool foldsAcross(ReduceOp red) const = 0;
};

const LayoutRules &tpuLayoutRules();

// The rules a pack names (its LAYOUT_RULES), or null when no rules have that name, and the names
// there are.
const LayoutRules *layoutRulesNamed(StringRef name);
std::string knownLayoutRules();

}
