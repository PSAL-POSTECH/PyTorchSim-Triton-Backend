#ifndef PYTORCHSIM_EXTERN_OPS
#define PYTORCHSIM_EXTERN_OPS

include "PyTorchSim/PyTorchSimDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// What a kernel states for itself through tl.extra.pytorchsim: convert-pytorchsim-extern-to-vcix writes a
// vcix on tiles as VCIX, once per register of each lane's rows.
def PyTorchSimExtern_Dialect : Dialect {
  let name = "pytorchsim_extern";
  let summary = "The ops a kernel writes itself.";
  let cppNamespace = "::pytorchsim::external";
}

class PyTorchSimExtern_Op<string mnemonic, list<Trait> traits = []>
    : Op<PyTorchSimExtern_Dialect, mnemonic, traits>;

def PyTorchSimExtern_VcixOp : PyTorchSimExtern_Op<"vcix", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<ConditionallySpeculatable>]> {
  let summary = "A VCIX instruction a kernel wrote by its encoding, on whole tiles in its layout.";
  let description = [{
    `form` is the VCIX op form (iv, v.i, v.iv), `opcode` and `imm` its fields, passed through:
    nothing here knows what the instruction does. `layouts` (one per operand) and `out_layout`
    are the kernel's statement of the TritonGPU encoding each tile takes; the anchor pins the
    types to them and nothing after reads them. One with no result or no operand reads and writes
    the state of the unit it names. One with both is assumed only to be a function of its
    operands -- nothing about what it computes. On tensors a result is the tile it yields; on
    buffers `dst` is the tile it writes.
  }];
  let arguments = (ins Variadic<AnyType>:$args, Optional<AnyMemRef>:$dst, I64Attr:$opcode,
                       I64Attr:$imm, StrAttr:$form, ArrayAttr:$layouts,
                       OptionalAttr<AnyAttr>:$out_layout);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Whether it reads or writes its unit's state: it has no result, or no operand.
    bool isStateful();
  }];
}

def PyTorchSimExtern_LayoutConstraintOp : PyTorchSimExtern_Op<"layout_constraint",
    [Pure, SameOperandsAndResultType, PyTorchSim_LayoutAnchor]> {
  let summary = "A tile the kernel or its target keeps in the layout it names; its value passes through.";
  let description = [{
    `layout` is a #ttg.blocked encoding with the machine's lanes in one warp and one CTA. The anchor pass
    gives the tile that encoding, the layout passes propagate from it, and strip-pins removes it.
  }];
  let arguments = (ins AnyRankedTensor:$src, AnyAttr:$layout);
  let results = (outs AnyRankedTensor:$result);
}

#endif
