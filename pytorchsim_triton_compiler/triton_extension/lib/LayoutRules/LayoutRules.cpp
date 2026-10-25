// The LayoutRules by the names packs give them (their LAYOUT_RULES): a machine's rules, one entry
// each.
#include "TorchsimExt/LayoutRules.h"

namespace mlir::triton::gpu {

namespace {

struct Named {
  StringRef name;
  const LayoutRules &(*rules)();
};

const Named kRules[] = {{"tpu", tpuLayoutRules}, {"rvv_generic", rvvGenericLayoutRules}};

}

const LayoutRules *layoutRulesNamed(StringRef name) {
  for (const Named &n : kRules)
    if (n.name == name)
      return &n.rules();
  return nullptr;
}

std::string knownLayoutRules() {
  std::string out;
  for (const Named &n : kRules)
    out += (out.empty() ? "" : ", ") + n.name.str();
  return out;
}

}
