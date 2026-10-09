"""A gather address built from an arange alone -- nested modulo -- reads wrong.

NOT DIAGNOSED. This file records a failure, it does not explain it. Found while
reducing DeepSeek-V3's RoPE kernel for
tests/cases/mechanism/gather/ptr_gather_i64_index_two_readers.py: the intervening
gather there was written in this shape and failed on its own, so it had to come
out of that kernel to keep the two apart. Measured FAIL on develop f6232e1 AND
with the index-buffer fix applied -- identical 127/128 elements outside
tolerance both ways -- so it predates that change and is not caused by it.

SIBLING: tests/cases/mechanism/gather/ptr_gather_modulo.py is the modulo case where
the expression ALSO involves a value loaded from memory, which is what forces
PtrAnalysis to give up the wrap-around `shape` encoding and rebuild the dim as a
gather. This file has NO loaded index -- every term comes from tl.arange -- and
it still ends up a gather.

WHAT IT LOWERS TO (07-regularized.mlir)
  tts.make_gather_scatter_tptr, gather_scatter_dim 0, strides [1], with the
  whole address materialised as linalg arithmetic on linalg.index:

      %13 = linalg.index 0                       r
      %14 = remsi %13, %12                       r % 64
      %16 = remsi %14, %11                       (r % 64) % 32
      %17 = muli  %16, %10                       * 2
      %18 = muli  %13, %9                        r * ?
      %19 = addi  %17, %18
      %20 = muli  %13, %8                        r * ?
      %21 = addi  %19, %20                       -> the gather offset

  TWO `r * const` terms for a source that has ONE (`576 * r`), and the `512 +`
  is nowhere in the offset -- presumably folded into the base. Whether that
  split is the bug or a red herring is exactly what is not established here.

WHAT IS ESTABLISHED
  - it is not the shared-index-buffer bug: one gather, one index tile, and the
    fix for that one changes nothing here
  - it is not the i64 route: this tile is i32 and takes the widening path
  - 127 of 128 elements are wrong, so it is not a boundary or a tail

WHY IT MATTERS NOW: DeepSeek-V3's RoPE kernel reads its q/k rows with
`512 + 2*((x0 % 32)) + 576*x1 + (x0 // 32)`, the same shape. Fixing the index
buffer got that kernel past its segfault; this is the next thing in its way.

WHERE THE FIX IS, found after this file was written. triton_shared's branch
`gather-offset-elem-type` fixes it, and not as a side effect worth guessing at:
`0453e53 PtrAnalysis: flatten a mixed dim whatever the strides are` makes the
structured side's implicit index materialise whatever the strides are, which is
the same repair the two siblings below need. MEASURED with a triton-shared-opt
built from those two commits rebased onto the in-use tip 37e28a9 -- this kernel
PASSES, and the suite goes 283 -> 287 of 313 with the only verdicts that move
being this file, ptr_gather_i64_offset_scaled_merge,
ptr_gather_structured_term_dropped and dl/ms_deform_attn, all FAIL -> PASS.
Not merged to `torchsim` yet, so it still fails on the pinned binary.

torch equivalent: x[512 + 2*((r % 64) % 32) + 576*r].

gather address from arange + nested modulo, no loaded index: wrong values on develop f6232e1, cause not established.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 73728
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_nested_modulo_of_arange_kernel(x_ptr, o_ptr,
                                                  BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        c = r % 64
        tl.store(o_ptr + r, tl.load(x_ptr + (512 + 2 * (c % 32) + 576 * r)))

    return ptr_gather_nested_modulo_of_arange_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    r = torch.arange(BLOCK)
    c = r % 64
    return {"o": x[512 + 2 * (c % 32) + 576 * r]}


SPEC = KernelSpec(
    name="ptr_gather_nested_modulo_of_arange_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (NX,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
