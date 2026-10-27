#ifndef PYTORCHSIM_DIALECT
#define PYTORCHSIM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PyTorchSim_Dialect : Dialect {
  let name = "pytorchsim";
  let summary = "The PyTorchSim NPU's own ops and markers: what every target lowers.";
  let cppNamespace = "::pytorchsim";
  let useDefaultAttributePrinterParser = 1;
}

class PyTorchSim_Op<string mnemonic, list<Trait> traits = []>
    : Op<PyTorchSim_Dialect, mnemonic, traits>;

// Resources and traits every dialect here shares: each .td of a dialect (PyTorchSimOps, PyTorchSimExternOps,
// TogsimOps, a target's) includes this file for them.
def PyTorchSim_Semaphore : Resource<"::pytorchsim::Semaphore">;
def PyTorchSim_InFlight : Resource<"::pytorchsim::InFlight">;
def PyTorchSim_RuntimeAttribute : Resource<"::pytorchsim::RuntimeAttribute">;

// An op whose result differs between lanes with no operand or bank to say so: the lane number.
def PyTorchSim_LaneVariant : NativeOpTrait<"LaneVariant"> { let cppNamespace = "::pytorchsim::OpTrait"; }
// A view of its source as one lane's bank: its result type's strided layout, in the bank's own
// coordinates, is all it states; convert-pytorchsim-to-llvm lowers it as a memref.reinterpret_cast.
def PyTorchSim_BankView : NativeOpTrait<"BankView"> { let cppNamespace = "::pytorchsim::OpTrait"; }
// A bank view read as elements of another type in the same descriptor: lowered as a change of type.
def PyTorchSim_RetypedView : NativeOpTrait<"RetypedView"> { let cppNamespace = "::pytorchsim::OpTrait"; }
// An op whose result layout the TritonGPU layout passes keep and propagate from (remove-layout-conversions
// starts from it and never rematerializes it): a layout the kernel or the machine states.
def PyTorchSim_LayoutAnchor : NativeOpTrait<"LayoutAnchor"> { let cppNamespace = "::pytorchsim::OpTrait"; }

defvar PyTorchSim_Landing = MemoryEffects<[MemRead, MemWrite, MemRead<PyTorchSim_Semaphore>,
                                         MemWrite<PyTorchSim_Semaphore>]>;

#endif
