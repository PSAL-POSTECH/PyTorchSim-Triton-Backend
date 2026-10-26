#ifndef PYTORCHSIM_RVV_OPS
#define PYTORCHSIM_RVV_OPS

include "PyTorchSim/PyTorchSimDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// The ops of a machine that computes on RVV and keeps its tiles in per-lane banks of a scratchpad: what
// the generic RVV lowering (lowering/ext/rvv_kind) makes, in any pack that adopts it.
def PyTorchSimRvv_Dialect : Dialect {
  let name = "pytorchsim_rvv";
  let summary = "The RVV target kind's own ops.";
  let cppNamespace = "::pytorchsim::rvv";
}

class PyTorchSimRvv_Op<string mnemonic, list<Trait> traits = []>
    : Op<PyTorchSimRvv_Dialect, mnemonic, traits>;

def PyTorchSimRvv_BankViewOp : PyTorchSimRvv_Op<"bank_view",
    [Pure, PyTorchSim_BankView, DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "A buffer seen with the strides a lane's bank really has.";
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$view);
}

#endif
