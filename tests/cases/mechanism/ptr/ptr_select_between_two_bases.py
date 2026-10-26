"""`tl.where` between two tensors of POINTERS -- selecting between addresses.

Upstream MLIR's `populateElementwiseToLinalgConversionPatterns` (called from
TritonArithToLinalg.cpp:107) wraps any Elementwise op over tensors in a
linalg.generic, and does that for `arith.select` on !tt.ptr as readily as for
arith.addf on f32.  So this arrives at pytorchsim-convert-unsupported-pointers as

    %15 = linalg.generic ins(%9, %12, %14 : tensor<128xi1>,
                             tensor<128x!tt.ptr<f32>>, tensor<128x!tt.ptr<f32>>)
          { arith.select %in, %in_0, %in_1 : !tt.ptr<f32> }
          -> tensor<128x!tt.ptr<f32>>

WHY THE ARM IS ONLY A RETYPING
Selecting between two addresses is selecting between two integers: the op
yields ONE OF ITS OWN INPUTS, unchanged, so it computes the same function at
i64.  And by the time the walk reaches it those inputs are ALREADY i64 -- convert_unsupported_pointers
runs in program order and the two tt.addptr generics feeding it sit above it,
so `replace_all_uses_with` has swapped them.  Only the result type and the
block argument types are still spelled !tt.ptr.

WHY `_ADDRESS_AGNOSTIC` IS A NAMED SET AND NOT A STRUCTURAL READ
No trait states this property, and the obvious structural test does not
separate it: `tt.addptr` ALSO takes a !tt.ptr and returns the same type, and
it is exactly the op the arm must not admit -- its index has to be scaled by
the element size first, which is why it has an arm of its own.

WHAT BREAKS THIS KERNEL
Deleting the `arith.select` entry from `_ADDRESS_AGNOSTIC`, which puts it back
on the refusal.  Measured on the tree before the arm landed:

    adapt: _Fatal: a linalg.generic over a tensor of pointers whose body is
    `arith.select` has no arm in this pass

Alternating the two bases per element is what makes the answer checkable: a
select that always took one side would agree with `a` everywhere and the
kernel would pass with the condition ignored.

Coverage: the guarded `else` in pytorchsim-convert-unsupported-pointers names the op it cannot dissolve,
instead of letting the next op report a chain that was not dissolved.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def select_between_two_bases_kernel(a_ptr, b_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        # tl.where over POINTERS: arith.select on !tt.ptr, wrapped in a generic.
        p = tl.where((offs % 2) == 0, a_ptr + offs, b_ptr + offs)
        tl.store(o_ptr + offs, tl.load(p))

    return select_between_two_bases_kernel


def make_inputs(torch, seed=0):
    return {"a": torch.arange(N, dtype=torch.float32),
            "b": -torch.arange(N, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(N, dtype=torch.float32)
    return {"o": torch.where(torch.arange(N) % 2 == 0, i, -i)}


SPEC = KernelSpec(
    name="select_between_two_bases_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": N},
    args=[Arg("a", IN, "float32", (N,)), Arg("b", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
