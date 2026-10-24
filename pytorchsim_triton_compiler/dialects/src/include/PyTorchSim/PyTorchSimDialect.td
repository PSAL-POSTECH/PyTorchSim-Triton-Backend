#ifndef TORCHSIM_DIALECT
#define TORCHSIM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PyTorchSim_Dialect : Dialect {
  let name = "pytorchsim";
  let summary = "The PyTorchSim NPU's own ops and markers: what every target lowers.";
  let cppNamespace = "::pytorchsim";
}

class PyTorchSim_Op<string mnemonic, list<Trait> traits = []>
    : Op<PyTorchSim_Dialect, mnemonic, traits>;

// Resources and traits every dialect here shares (TogsimOps.td includes this file for them).
def PyTorchSim_Semaphore : Resource<"::pytorchsim::Semaphore">;
def PyTorchSim_InFlight : Resource<"::pytorchsim::InFlight">;
def PyTorchSim_RuntimeAttribute : Resource<"::pytorchsim::RuntimeAttribute">;

// An op whose result differs between lanes with no operand or bank to say so: the lane number.
def PyTorchSim_LaneVariant : NativeOpTrait<"LaneVariant"> { let cppNamespace = "::pytorchsim::OpTrait"; }

defvar PyTorchSim_Landing = MemoryEffects<[MemRead, MemWrite, MemRead<PyTorchSim_Semaphore>,
                                         MemWrite<PyTorchSim_Semaphore>]>;

#endif
