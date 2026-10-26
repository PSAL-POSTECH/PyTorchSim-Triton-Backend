"""A constant offset on an index whose x axis is split by `%` and `//`.

    x1 = xindex % 64      the split axis's inner half
    x2 = xindex // 64     its outer half
    load  p + (C + x1 + S1*x2 + S2*y0)      C != 0

WHAT IT PINS, and every term of it was measured rather than guessed. Starting
from DeepSeek-V4's first divergent kernel and cutting it down, one load and one
store is enough -- no arithmetic at all -- and the failure needs exactly three
things at once:

    FAIL   C + x1 + S1*x2 + S2*y0       C = 1, 64 or 448 alike
    PASS       x1 + S1*x2 + S2*y0       no constant
    PASS   C +      S1*x2 + S2*y0       no inner half
    PASS   C + x1          + S2*y0       no outer half
    PASS   C + x1 + S1*x2               no second axis

So it is not the constant, not the split, and not the second axis -- it is the
three together. The value of the constant does not matter, which is what says
this is an address FOLD rather than an arithmetic slip.

WHERE IT CAME FROM. DeepSeek-V4-Flash-Base under transformers 5.15.0. The
per-kernel cross-check names buffer buf13 (aten.add.Tensor, 8187 of 8192
elements over tolerance) while its inputs pass, and the same kernel run in the
Triton INTERPRETER on the same runtime/*.raw inputs matches the CPU golden to
1.67e-06 -- so the kernel source says one thing and this backend does another.
Reduced from there, the arithmetic fell away and this address remained.

    tests/cases/mechanism/gather/ptr_gather_1d_arith.py is the closest existing
    kernel and it passes: one axis, no split.

WHERE THE FIX IS. The linalg lowering's PtrState::rebuildAsGatherScatter, which zeroed
a non-gather dim's offset along with its stride and shape. b6c3e60 on
pytorchsim-merge-develop already keeps a non-zero constant there -- it was written
from a collapsed max_pool2d window, whose constants sit on a dim that is NOT
split. This kernel reaches the same rule through the split, which that one does
not cover; measured FAIL at 5.82 against a build without it.

WHAT BREAKS IT AGAIN: making the constant zero, collapsing the split (x1 and x2
from separate program axes rather than one), or dropping to a single strided
axis. Each of those was measured to pass and each of them hides this.

A non-zero constant offset on an index whose x axis is split by %/// and which carries a second strided axis. Reduced from DeepSeek-V4's first divergent kernel; a load and a store, no arithmetic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS = 32          # the y axis
INNER = 64         # x1 = xindex % INNER
OUTER = 4          # x2 = xindex // INNER, so xnumel = INNER * OUTER
S1, S2 = 512, 2048  # the two strides V4's kernel uses
CONST = 448        # any non-zero constant reproduces it
NUMEL = CONST + (INNER - 1) + S1 * (OUTER - 1) + S2 * (ROWS - 1) + 1


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_const_offset_with_split_axis_kernel(
            in_ptr, out_ptr, ROWS: tl.constexpr, INNER: tl.constexpr,
            OUTER: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
            CONST: tl.constexpr):
        y0 = tl.arange(0, ROWS)[:, None]
        xindex = tl.arange(0, INNER * OUTER)[None, :]
        # ONE program axis, split into two by % and //. That split is what the
        # backend has to carry the constant through.
        x1 = xindex % INNER
        x2 = xindex // INNER

        v = tl.load(in_ptr + (CONST + x1 + S1 * x2 + S2 * y0))
        tl.store(out_ptr + (x1 + INNER * x2 + (INNER * OUTER) * y0), v)

    return ptr_const_offset_with_split_axis_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"src": torch.randn(NUMEL, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    src = inputs["src"]
    y = torch.arange(ROWS)[:, None]
    xi = torch.arange(INNER * OUTER)[None, :]
    idx = CONST + (xi % INNER) + S1 * (xi // INNER) + S2 * y
    return {"out": src[idx].reshape(ROWS * INNER * OUTER)}


SPEC = KernelSpec(
    name="ptr_const_offset_with_split_axis_kernel", kernel=kernel,
    signature={"in_ptr": "*fp32", "out_ptr": "*fp32", "ROWS": "constexpr",
               "INNER": "constexpr", "OUTER": "constexpr", "S1": "constexpr",
               "S2": "constexpr", "CONST": "constexpr"},
    constexprs={"ROWS": ROWS, "INNER": INNER, "OUTER": OUTER, "S1": S1,
                "S2": S2, "CONST": CONST},
    args=[Arg("src", IN, "float32", (NUMEL,)),
          Arg("out", OUT, "float32", (ROWS * INNER * OUTER,))],
    grid=(1,),
    reference=reference, make_inputs=make_inputs,
)
