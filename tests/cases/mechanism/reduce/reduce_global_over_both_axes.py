"""A GLOBAL reduction -- `tl.sum(x)` with no axis -- over a rank-2 tile.

    x    (M, N)
    row  tl.max(x, axis=1)   a SECOND consumer, so x keeps its rank
    s    tl.sum(x)           linalg.reduce dimensions = [0, 1] -> tensor<f32>
    out  x - row[:, None] - s / (M*N)

`tl.sum(x)` ON ITS OWN is flattened to a rank-1 tile and one `dimensions = [0]`
reduce, which is why the second consumer is here: with the rank pinned, the
global reduction arrives as the two-axis fold it really is.

WHAT THIS PINS. The machine banks ONE axis, so a reduce that names two says
nothing about which; `_split_reduces_over_two_axes` turns it into two reduces of
one axis each, above the pass that has to choose. That split needs the identity
the accumulator was seeded with, so the intermediate can start from the same
one -- and it looked for a `linalg.fill`.

A GLOBAL REDUCTION HAS NO FILL. Its accumulator is rank 0, and the linalg lowering
seeds a rank-0 tensor by writing into it:

    %6 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %6[] : tensor<f32>

so the seeding was never found, the split declined, and the two-axis reduce went
down whole to be refused five passes later:

    NotVectorisable: reduce over [0, 1] of a tile that holds more than one axis
    on the lanes

Every global reduction in the language was outside the split for that one
reason, and no kernel in the suite had one to say so.

WHAT WOULD BREAK THIS. Narrowing `_identity_seeding` back to `linalg.fill`, or
accepting an insert into a tensor that is not EMPTY -- then the split would
start an intermediate from a value that already carries data.

A rank-0 accumulator is seeded by tensor.insert, not linalg.fill, which kept every global reduction out of the two-axis split.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 512, 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def global_over_both_axes(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
        # THE SECOND CONSUMER, which keeps x rank 2 so the global sum below
        # cannot be flattened into a rank-1 reduce.
        row = tl.max(x, axis=1)
        # NO AXIS. One `linalg.reduce` over dimensions = [0, 1].
        s = tl.sum(x) / (M * N)
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], x - row[:, None] - s)

    return global_over_both_axes


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, N)
    o = x - x.max(dim=1, keepdim=True).values - x.mean()
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="global_over_both_axes", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)),
          Arg("o", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
