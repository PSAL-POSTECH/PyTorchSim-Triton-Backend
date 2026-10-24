#ifndef TORCHSIM_TORCHSIMOPS_H
#define TORCHSIM_TORCHSIMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace pytorchsim {
/// The memref memory space that is the scratchpad.
constexpr int64_t kScratchpadSpace = 1;
/// The state a VCIX unit keeps between instructions: what a push adds and a pop takes. No
/// buffer is in it, so an effect on it is not an access to any memref.
struct UnitState : public mlir::SideEffects::Resource::Base<UnitState> {
  llvm::StringRef getName() const final { return "pytorchsim::UnitState"; }
  bool isAddressable() const final { return false; }
};
/// The semaphores transfers signal and waits and barriers consume; not a buffer either.
struct Semaphore : public mlir::SideEffects::Resource::Base<Semaphore> {
  llvm::StringRef getName() const final { return "pytorchsim::Semaphore"; }
  bool isAddressable() const final { return false; }
};
/// The transfers in flight into or out of one buffer, which a wait on them reads and writes: an
/// effect on it names the buffer and is not an access to any of its elements.
struct InFlight : public mlir::SideEffects::Resource::Base<InFlight> {
  llvm::StringRef getName() const final { return "pytorchsim::InFlight"; }
};
/// What a runtime attribute record writes and its read point reads: the run's record of a value
/// the TOG takes, not memory of the program.
struct RuntimeAttribute : public mlir::SideEffects::Resource::Base<RuntimeAttribute> {
  llvm::StringRef getName() const final { return "pytorchsim::RuntimeAttribute"; }
  bool isAddressable() const final { return false; }
};
namespace OpTrait {
/// An op whose result differs between lanes though no operand and no bank it reads says so.
template <typename C> class LaneVariant : public mlir::OpTrait::TraitBase<C, LaneVariant> {};
}
}

#define GET_OP_CLASSES
#include "PyTorchSim/PyTorchSimOps.h.inc"

#endif
