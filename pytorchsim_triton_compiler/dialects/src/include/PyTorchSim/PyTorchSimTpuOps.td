#ifndef PYTORCHSIM_TPU_OPS
#define PYTORCHSIM_TPU_OPS

include "PyTorchSim/PyTorchSimDialect.td"

// The TPU family's own ops: what its pack (lowering/ext/pytorchsim_tpu_family) makes inside a slot and
// lowers itself, in lower-target-ops. No common pass makes or reads one.
def PyTorchSimTpu_Dialect : Dialect {
  let name = "pytorchsim_tpu";
  let summary = "The TPU family's units' own ops.";
  let cppNamespace = "::pytorchsim::tpu";
}

class PyTorchSimTpu_Op<string mnemonic, list<Trait> traits = []>
    : Op<PyTorchSimTpu_Dialect, mnemonic, traits>;

def PyTorchSimTpu_XluMoveOp : PyTorchSimTpu_Op<"xlu_move"> {
  let summary = "One tile carried into another's lanes by the cross-lane unit (XLU).";
  let description = [{
    `xlu_op` is the XLU op (its SIMM5), `kind` how the pack emits it (crossing, gather or permute)
    and `rounds` the plan it was priced at. `every_lane`: every lane takes each destination element
    from the same source element, so a UNIFORM destination stays uniform (check-lane-uniformity
    reads it as the marker pytorchsim.every_lane). Buffers only: lower-convert makes it from a
    pytorchsim.convert_layout once every buffer is placed, so it never meets a tensor.
  }];
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$src, Arg<AnyMemRef, "", [MemWrite]>:$dst,
                       I64Attr:$xlu_op, StrAttr:$kind, OptionalAttr<ArrayAttr>:$rounds);
}

#endif
