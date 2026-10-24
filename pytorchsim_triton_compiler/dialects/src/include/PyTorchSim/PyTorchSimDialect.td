#ifndef TORCHSIM_DIALECT
#define TORCHSIM_DIALECT

include "mlir/IR/OpBase.td"

def PyTorchSim_Dialect : Dialect {
  let name = "pytorchsim";
  let summary = "The PyTorchSim NPU's own ops and markers.";
  let cppNamespace = "::pytorchsim";
}

class PyTorchSim_Op<string mnemonic, list<Trait> traits = []>
    : Op<PyTorchSim_Dialect, mnemonic, traits>;

#endif
