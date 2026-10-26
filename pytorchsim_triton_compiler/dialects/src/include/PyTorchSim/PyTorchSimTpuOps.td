#ifndef PYTORCHSIM_TPU_OPS
#define PYTORCHSIM_TPU_OPS

include "PyTorchSim/PyTorchSimDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// The TPU kind's own ops: what its passes (lowering/ext/tpu_kind) make
// (convert-pytorchsim-to-target-kind), written as VCIX by convert-to-vcix. No common pass makes or reads one.
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
    `mode` is the move (permute, gather or transpose: the xlu.push / xlu.pop mode it lowers to) and
    `rounds` the plan it was priced at. Buffers only: lower-convert makes it from a
    pytorchsim.convert_layout once every buffer is placed, so it never meets a tensor.

    `pytorchsim.every_lane`: every lane takes each destination element from the same source
    element, so a UNIFORM destination stays uniform (verify-lane-uniformity reads it).
  }];
  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$src, Arg<AnyMemRef, "", [MemWrite]>:$dst,
                       StrAttr:$mode, OptionalAttr<ArrayAttr>:$rounds);
}

def PyTorchSimTpu_VpuBankViewOp : PyTorchSimTpu_Op<"vpu.bank_view",
    [Pure, PyTorchSim_BankView, DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "A buffer seen with the strides a lane's bank of the VPU's scratchpad really has.";
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$view);
}

def PyTorchSimTpu_VpuBankWordsOp : PyTorchSimTpu_Op<"vpu.bank_words",
    [Pure, PyTorchSim_BankWords, DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "A lane's bank read as words of 32 bits, as many of its elements to a word as fit.";
  let arguments = (ins AnyMemRef:$bank);
  let results = (outs AnyMemRef:$words);
}

def PyTorchSimTpu_MiscLaneIdxOp : PyTorchSimTpu_Op<"misc.lane_idx"> {
  let summary = "Each lane's own number, read from the Misc unit (vlane_idx), `vl` elements of it.";
  let arguments = (ins AnyTypeOf<[I32, I64]>:$vl);
  let results = (outs AnyVectorOfAnyRank:$idx);
}

def PyTorchSimTpu_MiscPackOp : PyTorchSimTpu_Op<"misc.pack"> {
  let summary = "bf16 of `lo`'s f32 then `hi`'s, rounded to nearest even, two to a word (vpack).";
  let arguments = (ins AnyVectorOfAnyRank:$lo, AnyVectorOfAnyRank:$hi, Optional<AnyTypeOf<[I32, I64]>>:$vl);
  let results = (outs AnyVectorOfAnyRank:$words);
}

def PyTorchSimTpu_MiscUnpackLoOp : PyTorchSimTpu_Op<"misc.unpack_lo"> {
  let summary = "The first `vl` bf16 of `words`' 2 * vl, widened to f32.";
  let arguments = (ins AnyVectorOfAnyRank:$words, AnyTypeOf<[I32, I64]>:$vl);
  let results = (outs AnyVectorOfAnyRank:$wide);
}

def PyTorchSimTpu_MiscUnpackHiOp : PyTorchSimTpu_Op<"misc.unpack_hi"> {
  let summary = "The last `vl` bf16 of `words`' 2 * vl, widened to f32.";
  let arguments = (ins AnyVectorOfAnyRank:$words, AnyTypeOf<[I32, I64]>:$vl);
  let results = (outs AnyVectorOfAnyRank:$wide);
}

// One elementwise function of the SFU each.
class PyTorchSimTpu_SfuOp<string fn> : PyTorchSimTpu_Op<"sfu." # fn> {
  let summary = "Elementwise " # fn # " in the SFU.";
  let arguments = (ins AnyVectorOfAnyRank:$x);
  let results = (outs AnyVectorOfAnyRank:$y);
}
def PyTorchSimTpu_SfuExpOp : PyTorchSimTpu_SfuOp<"exp">;
def PyTorchSimTpu_SfuErfOp : PyTorchSimTpu_SfuOp<"erf">;
def PyTorchSimTpu_SfuTanhOp : PyTorchSimTpu_SfuOp<"tanh">;
def PyTorchSimTpu_SfuSinOp : PyTorchSimTpu_SfuOp<"sin">;
def PyTorchSimTpu_SfuCosOp : PyTorchSimTpu_SfuOp<"cos">;
def PyTorchSimTpu_SfuLogOp : PyTorchSimTpu_SfuOp<"log">;
def PyTorchSimTpu_SfuAtanOp : PyTorchSimTpu_SfuOp<"atan">;

def PyTorchSimTpu_EupPushOp : PyTorchSimTpu_Op<"eup.push"> {
  let summary = "Pushes `x` into the EUP, which computes 2^x; eup.pop takes the oldest push's result.";
  let arguments = (ins AnyVectorOfAnyRank:$x);
}

def PyTorchSimTpu_EupPopOp : PyTorchSimTpu_Op<"eup.pop"> {
  let summary = "Takes the oldest eup.push's result.";
  let results = (outs AnyVectorOfAnyRank:$y);
}

def PyTorchSimTpu_MxuPrepOp : PyTorchSimTpu_Op<"mxu.prep"> {
  let summary = "Holds one half of a register pair in the MXU (vmatprep): weights', or an input's.";
  let arguments = (ins AnyVectorOfAnyRank:$words, AnyTypeOf<[I32, I64]>:$vl, I64Attr:$format,
                       BoolAttr:$input);
}

def PyTorchSimTpu_MxuPushOp : PyTorchSimTpu_Op<"mxu.push"> {
  let summary = "Stages weights from the held half and `words` (vmatpush) into one half of the depth.";
  let arguments = (ins AnyVectorOfAnyRank:$words, AnyTypeOf<[I32, I64]>:$vl, I64Attr:$format,
                       BoolAttr:$second_half);
}

def PyTorchSimTpu_MxuMatmulOp : PyTorchSimTpu_Op<"mxu.matmul"> {
  let summary = "Streams one register of input rows (vmatmul); `latch` first makes staged weights active.";
  let arguments = (ins AnyVectorOfAnyRank:$rows, AnyTypeOf<[I32, I64]>:$vl, I64Attr:$format, BoolAttr:$latch);
}

def PyTorchSimTpu_MxuPopOp : PyTorchSimTpu_Op<"mxu.pop"> {
  let summary = "Takes one register of output rows (vpop): the lower column half, or the upper.";
  let arguments = (ins AnyTypeOf<[I32, I64]>:$vl, BoolAttr:$upper);
  let results = (outs AnyVectorOfAnyRank:$rows);
}

def PyTorchSimTpu_XluSetPatternOp : PyTorchSimTpu_Op<"xlu.set_pattern"> {
  let summary = "Sets the source lane of each lane for the permutes pushed after it (xlu_set_pattern).";
  let arguments = (ins AnyVectorOfAnyRank:$lanes, AnyTypeOf<[I32, I64]>:$vl);
}

def PyTorchSimTpu_XluPushOp : PyTorchSimTpu_Op<"xlu.push"> {
  let summary = "Pushes a register of words into the XLU; `mode` is permute, gather or transpose.";
  let arguments = (ins AnyVectorOfAnyRank:$words, AnyTypeOf<[I32, I64]>:$vl, StrAttr:$mode);
}

def PyTorchSimTpu_XluPushReduceOp : PyTorchSimTpu_Op<"xlu.push_reduce"> {
  let summary = "Pushes a register the XLU combines across the lanes by `kind` (max, sum or min).";
  let arguments = (ins AnyVectorOfAnyRank:$words, AnyTypeOf<[I32, I64]>:$vl, StrAttr:$kind);
}

def PyTorchSimTpu_XluPopOp : PyTorchSimTpu_Op<"xlu.pop"> {
  let summary = "Takes the oldest push's result; `mode` is its push's (permute, gather, transpose, reduce).";
  let arguments = (ins AnyTypeOf<[I32, I64]>:$vl, StrAttr:$mode);
  let results = (outs AnyVectorOfAnyRank:$words);
}

#endif
