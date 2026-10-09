"""A reduction fed by two layouts gives a lane axis one element deep.

    a[x, r] = A[r + R*x]        row-major
    b[x, r] = B[x + X*r]        column-major -- a NON-unit stride on r
    o[x]    = sum(a + b, axis=1) / R

WHAT THIS PINS. vectorize retypes a buffer to the strides one lane's bank
really has, and a lane axis holding ONE element per lane does not move, so its
stride is zero. `_bank_order` sorts unit axes AHEAD of the rest, so that zero
lands in the FIRST slot rather than the last:

    memref<1x1x64xf32, strided<[0, 64, 1]>, 1>

which no MLIR parser will read back -- StridedLayoutAttr verifies "strides must
not be zero". Every pass in this repository builds and rewrites the type through
the bindings and never re-parses it, so the whole pipeline runs to completion and
the refusal arrives one stage later, in whoever reads the printed IR. For
PyTorchSim that is the TOG builder:

    build_tog.run_tog -> ir.Module.parse(09-custom.mlir)
    MLIRError: error: "-":113:158: strides must not be zero

The trailing-zero case was already handled, because sorting by descending stride
puts a zero last and vector.load rejects a most-minor stride that is not 1
(layernorm, bert_embeddings, convnext_block, gqa_repeat_kv, groupnorm_film). The
leading one needs the lane axis to be one element deep, which is why this kernel
reduces rather than maps.

WHY BOTH LAYOUTS. With only the row-major load every access folds into one
contiguous view and the retype never fires. The column-major operand is what
gives the packed view a second axis with a stride worth writing down, so the
buffer is rank 3 and the zero has somewhere to sit that is not the end.

resnet18 reaches this at its global average pool, whose batch-norm epilogue reads
one operand as [x, r] and the residual as [r, x].

A lane axis one element deep carries stride zero into the FIRST slot of a banked view's type, which MLIR will not parse back.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

X, R = 128, 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mean_of_two_layouts_kernel(a_ptr, b_ptr, o_ptr, X: tl.constexpr,
                                   R: tl.constexpr):
        x = tl.arange(0, X)[:, None]
        r = tl.arange(0, R)[None, :]
        a = tl.load(a_ptr + (r + R * x))          # row-major
        b = tl.load(b_ptr + (x + X * r))          # column-major
        s = tl.sum(a + b, 1)[:, None]
        tl.store(o_ptr + x, s / R)

    return mean_of_two_layouts_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(X * R, generator=g, dtype=torch.float32),
            "b": torch.randn(X * R, generator=g, dtype=torch.float32)}


def reference(inputs):
    a = inputs["a"].reshape(X, R)
    b = inputs["b"].reshape(R, X).t()
    return {"o": ((a + b).sum(1) / R).reshape(-1)}


SPEC = KernelSpec(
    name="mean_of_two_layouts_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "X": "constexpr", "R": "constexpr"},
    constexprs={"X": X, "R": R},
    args=[Arg("a", IN, "float32", (X * R,)),
          Arg("b", IN, "float32", (X * R,)),
          Arg("o", OUT, "float32", (X,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
