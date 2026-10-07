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

#endif
