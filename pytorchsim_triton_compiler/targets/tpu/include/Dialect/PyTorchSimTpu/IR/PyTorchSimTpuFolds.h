#ifndef PYTORCHSIM_TPU_FOLDS_H
#define PYTORCHSIM_TPU_FOLDS_H

#include "PyTorchSim/CrossLaneFolds.h"

namespace pytorchsim::tpu {
/// What the XLU's push_reduce folds across the lanes (targets/tpu/CMakeLists.txt names it).
pytorchsim::CrossLaneFolds crossLaneFolds();
}

#endif
