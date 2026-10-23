// What one op does to one buffer, as MemoryEffectOpInterface states it, and the buffer a view is
// of, as ViewLikeOpInterface states it: the questions the passes in Python ask through _torchsim_ext
// so that no pass answers them from an op's name.
#include "TorchSim-c/Dialects.h"
#include "TorchSim/TorchSimOps.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;

/// `effects` as TORCHSIM_EFFECT_* bits: one on the transfers in flight (torchsim::InFlight) is
/// SYNC, not a read or write of a buffer's data.
static int effectBits(ArrayRef<MemoryEffects::EffectInstance> effects) {
  int got = 0;
  for (const MemoryEffects::EffectInstance &e : effects) {
    if (e.getResource() == torchsim::InFlight::get()) {
      got |= TORCHSIM_EFFECT_SYNC;
      continue;
    }
    MemoryEffects::Effect *k = e.getEffect();
    got |= isa<MemoryEffects::Read>(k)       ? TORCHSIM_EFFECT_READ
           : isa<MemoryEffects::Write>(k)    ? TORCHSIM_EFFECT_WRITE
           : isa<MemoryEffects::Allocate>(k) ? TORCHSIM_EFFECT_ALLOCATE
                                             : TORCHSIM_EFFECT_FREE;
  }
  return got;
}

/// The effects `op` states on `v` as TORCHSIM_EFFECT_* bits; one on the transfers in flight there
/// (torchsim::InFlight) is SYNC, not a read or write of the buffer's data. -1 when `op` states no
/// effects at all, so nothing can be said.
extern "C" int torchsimEffectsOn(MlirOperation c_op, MlirValue c_v) {
  Operation *op = unwrap(c_op);
  Value v = unwrap(c_v);
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return TORCHSIM_EFFECT_UNKNOWN;
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffectsOnValue(v, effects);
  return effectBits(effects);
}

/// Every effect `op` states, on any value or resource, its regions' included, as
/// TORCHSIM_EFFECT_* bits; -1 when it or an op in it states no effects at all.
extern "C" int torchsimEffectsOf(MlirOperation c_op) {
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(unwrap(c_op));
  return effects ? effectBits(*effects) : TORCHSIM_EFFECT_UNKNOWN;
}

/// The value `v` is a view of, through every ViewLikeOpInterface whose source is in the same
/// memory space: a cast into another space is another buffer's name, not a view of this one.
extern "C" MlirValue torchsimViewRoot(MlirValue c_v) {
  Value v = unwrap(c_v);
  while (auto view = v.getDefiningOp<ViewLikeOpInterface>()) {
    Value src = view.getViewSource();
    auto to = dyn_cast<BaseMemRefType>(v.getType());
    auto from = dyn_cast<BaseMemRefType>(src.getType());
    if (view.getViewDest() != v || !to || !from || to.getMemorySpace() != from.getMemorySpace())
      break;
    v = src;
  }
  return wrap(v);
}

/// Whether `op` declares torchsim::OpTrait::LaneVariant: its result differs between lanes.
extern "C" bool torchsimIsLaneVariant(MlirOperation c_op) {
  return unwrap(c_op)->hasTrait<torchsim::OpTrait::LaneVariant>();
}

/// Whether `op` states no memory effect at all, on any value or resource, its regions' included.
extern "C" bool torchsimIsMemoryEffectFree(MlirOperation c_op) {
  return isMemoryEffectFree(unwrap(c_op));
}
