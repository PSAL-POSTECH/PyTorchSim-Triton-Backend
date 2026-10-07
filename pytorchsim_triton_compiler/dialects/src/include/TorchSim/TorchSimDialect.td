//===- TorchSimDialect.td - what this backend stamps on the IR -------------===//
//
// `torchsim` was an ATTRIBUTE NAMESPACE carried by unregistered ops until this
// dialect. Registering it buys a verifier for schemas that were held by comment
// alone, and is where bufferization learns what a transfer reads and writes.
//
//===----------------------------------------------------------------------===//
#ifndef TORCHSIM_DIALECT
#define TORCHSIM_DIALECT

include "mlir/IR/OpBase.td"

def TorchSim_Dialect : Dialect {
  let name = "torchsim";
  let summary = "The PyTorchSim NPU's own ops and markers.";
  let cppNamespace = "::torchsim";
}

class TorchSim_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchSim_Dialect, mnemonic, traits>;

#endif // TORCHSIM_DIALECT
