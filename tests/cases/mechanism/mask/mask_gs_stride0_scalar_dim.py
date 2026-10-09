"""A scalar parked on a STRIDE-0 dim, then rebuilt as a gather: divide by zero.

yolo11's PSA epilogue shape. `idx_y_c[None, :] + group * C` parks the group term
on the size-1, stride-0 dim -- where visitOperandSplat puts a splatted scalar --
and an unstructured mask sends the pointer through applyUnstructuredMask, whose
conversion divides the offset by that dim's stride.

WHAT IT PINS, measured: with the stride-0 arm removed the answer is 48 off; with
it, exact. THE GRID IS WHY IT IS FOUR PROGRAMS AND NOT ONE -- at group 0 the
parked term is zero, `0 / 0` comes out 0, and the kernel passes while proving
nothing. That is how the first version of this file failed to pin anything.

THAT CLAIM IS NO LONGER THIS FILE'S, and it did not move because anything about
it was wrong. 3115d94 sends a READ's unstructured mask to an arith.select
instead of rebuilding the pointer, so the load above stopped reaching
applyUnstructuredMask at all and this file went vacuous in place. Measured
2026-08-28: with the arm removed the load form still PASSED, and putting
`unstructured-mask-as-select=false` back brought the 48 straight back.

A write still demotes, so the mask moved to a store -- and the store is what
this file pins NOW, because the arm's own evidence could not follow it there.
The two trees differ only in an offset LIST (`offsets: [0, %term]` against
`offsets: [0, 0]`), which expect_ir cannot see: it counts op names, and every
count is equal. That witness is a lit test where the code is, at
triton_shared test/Conversion/TritonToStructured/scalar_not_parked_on_stride0.mlir.

WHAT THIS FILE PINS NOW is that the mask never becomes a gather at all. It is
one runtime boolean -- `um[:, None]` broadcast over the row gives every element
the same bool -- so it cannot VARY along an axis that has one position. Recording
it as an unstructured mask on that axis forced a gather axis of stride 0, which
is the one axis a gather cannot use: the rebuild divides that axis's offset by
its stride, and a masked scatter has to park its masked-off offsets by stepping
that stride. Both are the same impossibility, and both went away when the mask
became what it is -- the axis's EXTENT, 1 when true and 0 when false:

    %36 = arith.select %um, %c1, %c0 : index
    "tts.store"(%34, %24, %36) <{static_mask_dims = array<i64: -9223372036854775808, 16>}>

no gather, no scatter, no divide. expect_ir below holds that: the whole kernel
comes out on tts.make_tptr.

The suite reached this arm ZERO times before this file: instrumented over 389
kernels, the gather rebuild is entered 8 times with strides 16, 1 and 128, and
never 0.

the only kernel whose unstructured mask sits on an axis of one position. It must fold to that axis's runtime extent, not to a gather on a stride-0 axis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, C, GROUPS = 8, 16, 4


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_gs_stride0_scalar_dim_kernel(x_ptr, beta_ptr, m_ptr, o_ptr, s_ptr,
                               ROWS: tl.constexpr, C: tl.constexpr):
        group = tl.program_id(0)
        r = tl.arange(0, ROWS)
        idx_y_c = tl.arange(0, C)
        um = tl.load(m_ptr + tl.arange(0, 1)) < 5          # data-dependent
        idx_c = idx_y_c[None, :] + group * C               # group on dim 0
        beta = tl.load(beta_ptr + idx_c)
        base = group * (ROWS * C)
        v = tl.load(x_ptr + (base + r[:, None] * C + idx_y_c[None, :]))
        tl.store(o_ptr + (base + r[:, None] * C + idx_y_c[None, :]), v + beta)
        tl.store(s_ptr + idx_c, beta, mask=um[:, None])

    return mask_gs_stride0_scalar_dim_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(GROUPS * ROWS * C, generator=g, dtype=torch.float32),
            "beta": torch.arange(100, 100 + GROUPS * C, dtype=torch.float32),
            "m": torch.zeros(1, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(GROUPS, ROWS, C)
    beta = inputs["beta"].reshape(GROUPS, 1, C)
    return {"o": (x + beta).reshape(-1), "s": inputs["beta"].clone()}


SPEC = KernelSpec(
    name="mask_gs_stride0_scalar_dim_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "beta_ptr": "*fp32", "m_ptr": "*fp32",
               "o_ptr": "*fp32", "s_ptr": "*fp32",
               "ROWS": "constexpr", "C": "constexpr"},
    constexprs={"ROWS": ROWS, "C": C},
    args=[Arg("x", IN, "float32", (GROUPS * ROWS * C,)),
          Arg("beta", IN, "float32", (GROUPS * C,)),
          Arg("m", IN, "float32", (1,)),
          Arg("o", OUT, "float32", (GROUPS * ROWS * C,)),
          Arg("s", OUT, "float32", (GROUPS * C,))],
    grid=(GROUPS,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 0,
                                    "tts.gather": 0, "tts.scatter": 0,
                                    "tts.make_tptr": 6}},
)
