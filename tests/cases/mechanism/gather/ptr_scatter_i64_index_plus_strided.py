"""Scatter to `o[i + 8*x1]` with i loaded as i64: the extension is dropped.

The store-side twin of
tests/cases/mechanism/gather/ptr_gather_i64_offset_scaled_merge.py, and the shape
DeepSeek-V3's MoE scatter has -- triton_npu_fused_scatter_zeros_like_39:

    tmp0 = tl.load(in_ptr0 + (x2), xmask)        # *i64, top-k's output
    tl.device_assert((0 <= tmp0) & (tmp0 < 8), ...)
    tl.store(out_ptr0 + (tmp0 + 8*x1), 1.0, xmask)

THIS ONE PASSES TODAY, and that is the point of the file: it is a guard, not a
report. It fails on triton_shared's `gather-offset-elem-type` -- the branch that
fixes this file's three gather siblings -- so whoever merges that branch has to
carry the extsi repair with it.

    82208e0 build the flattened offset in the offset tensor's own type
       -> this kernel PASSES, the two silently-wrong gathers stay wrong
    0453e53 flatten a mixed dim whatever the strides are
       -> the two gathers are fixed, THIS BREAKS:
          error: 'arith.addi' op requires the same type for all operands
          %22 = "arith.addi"(%18, %21)
                : (tensor<128xi64>, tensor<128xi32>) -> tensor<128xi64>
    + PtrAnalysis: keep the extension when visiting an extsi
       -> all three pass

WHAT IS ACTUALLY WRONG, since the middle commit is not the culprit. The incoming
Triton IR is well typed; triton extends the i32 term itself:

    %0 = arith.muli %x1_1, 8 : tensor<128xi32>
    %1 = arith.extsi %0      : tensor<128xi32> to tensor<128xi64>
    %2 = arith.addi %i_3, %1 : tensor<128xi64>

but `PtrAnalysis::visitOperandExtSI` visits the input and returns, so the state
keeps the NARROW tensor while standing for the WIDE value. While a mixed dim was
flattened only on the differing-stride arm there was no place that put the two
offsets side by side. Flattening every mixed dim is what makes the lie meet its
consequence. So the middle commit exposed a width loss that was already there --
it did not introduce one.

WHY THE SUITE DID NOT CATCH IT. 313 kernels, 287 passing, identical with and
without the repair: no kernel had a store whose address adds a loaded i64 index
to a strided arange. A shape with no witness cannot be counted as a regression,
which is the whole reason this file exists rather than a note somewhere.

torch equivalent: o = zeros(NO); o[i + 8*(arange(BLOCK)//4)] = 1.0.

store address = loaded i64 index + strided arange; guards the extsi widening that triton_shared's mixed-dim flatten depends on.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NO = 1024
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_scatter_i64_index_plus_strided_kernel(i_ptr, o_ptr,
                                                   BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x1 = r // 4
        i = tl.load(i_ptr + r)                   # i64, so the address is i64
        tl.store(o_ptr + (i + 8 * x1), 1.0)      # ... and 8*x1 is i32

    return ptr_scatter_i64_index_plus_strided_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"i": torch.randint(0, 8, (BLOCK,), generator=g,
                               dtype=torch.int64)}


def reference(inputs):
    import torch
    i = inputs["i"].long()
    r = torch.arange(BLOCK)
    o = torch.zeros(NO, dtype=torch.float32)
    o[i + 8 * (r // 4)] = 1.0
    return {"o": o}


SPEC = KernelSpec(
    name="ptr_scatter_i64_index_plus_strided_kernel", kernel=kernel,
    signature={"i_ptr": "*i64", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("i", IN, "int64", (BLOCK,)), Arg("o", OUT, "float32", (NO,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
