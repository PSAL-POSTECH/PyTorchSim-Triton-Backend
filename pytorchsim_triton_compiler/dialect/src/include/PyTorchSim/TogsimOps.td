#ifndef TOGSIM_OPS
#define TOGSIM_OPS

include "PyTorchSim/PyTorchSimDialect.td"

// The tile-operation graph as TOGSim's API has it (togsim.hpp): what togsim/ emits from a lowered
// kernel and lower_to_emitc writes as C++. Nothing of the kernel's own lowering is here.
def Togsim_Dialect : Dialect {
  let name = "togsim";
  let summary = "The TOG's ops: TOGSim API calls, one op each.";
  let cppNamespace = "::togsim";
}

class Togsim_Op<string mnemonic, list<Trait> traits = []>
    : Op<Togsim_Dialect, mnemonic, traits>;

def Togsim_DmaOp : Togsim_Op<"dma", [PyTorchSim_Landing]> {
  let summary = "A DMA of the tile-operation-graph (TOG): one togsim::load or togsim::store call, signaling `sem` when it is an async load.";
  let description = [{
    `extents` size the dims its `dyn_dims` attribute names at run time: what a mask leaves live.
  }];
  let arguments = (ins Index:$dram_idx, Index:$sem, Variadic<Index>:$extents);
}

def Togsim_ComputeOp : Togsim_Op<"compute", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "The TOG's compute: one togsim::compute call.";
}

def Togsim_AttributeReadOp : Togsim_Op<"attribute_read",
    [MemoryEffects<[MemRead<PyTorchSim_RuntimeAttribute>]>]> {
  let summary = "The TOG's read of runtime attribute `id`: one togsim::scalar, or togsim::element at `indices`.";
  let arguments = (ins Variadic<Index>:$indices, I32Attr:$id, I32Attr:$rank);
  let results = (outs AnySignlessIntegerOrIndex:$value);
}

def Togsim_MemoryBarrierOp : Togsim_Op<"memory_barrier", [PyTorchSim_Landing]> {
  let summary = "The TOG's barrier: one togsim::wait call, waiting every load that signaled `sem` since its last wait.";
  let arguments = (ins Index:$sem);
}

#endif
