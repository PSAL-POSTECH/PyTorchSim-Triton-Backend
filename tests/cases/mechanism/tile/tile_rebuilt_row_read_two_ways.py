"""ONE computed row read down BOTH axes of one tile, rebuilt for each.

    i = tl.arange(0, N)[:, None]      one memref<Nxi32>, read (d0, d1) -> (d0)
    j = tl.arange(0, N)[None, :]      the SAME one, read (d0, d1) -> (d1)
    o = x * (i - j)

WHAT IT PINS. `fit_layout_to_lanes._rebuild_computed_row` answers "this row has to
be present WHOLE in every bank" by building the row's producer at the tile's
shape -- free, because the producer reads nothing and every lane can replay it.
It then re-points the readers at the rebuilt buffer and gives each of them the
IDENTITY map, which is the right map for a reader whose own map was the one the
rebuild was made for and the WRONG one for any other.

Triton CSEs the two `tl.arange(0, N)` calls into ONE value, so after bufferize
there is a single `memref<64xi32>` with two uses in one `linalg.generic`, one
per axis. Retargeting both onto a single rebuild makes the op compute
`rk[i] - rk[i]`, which is zero everywhere: the output comes out equal to
0 * x = 0 instead of x * (i - j), with no diagnostic anywhere -- the IR
verifies, the DMA descriptors are consistent, and only the values say so.
Measured on dl/alibi, which is this shape with an exponent on top.

The rewrite therefore retargets only the uses whose map MATCHES, and leaves the
others reading the narrow row -- which forces the clone disposition rather than
the move, so the second use can be rebuilt in its own right on a later round.

WHAT MAKES IT FAIL AGAIN: retarget every use of the row, or compare readers by
anything other than the iteration dims their maps name. The failure is silent
and total (max_abs_err equals the reference's own magnitude), which is why the
reference is asymmetric in i and j -- a symmetric one would let `i - j` and
`j - i` both look plausible.

torch equivalent: o[i, j] = x[i, j] * (i - j).

A row rebuilt at tile shape may only take the readers whose maps name the same iteration dims; retargeting a use that reads the row down the OTHER axis makes the op compute rk[i] - rk[i] = 0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rebuilt_row_read_two_ways_kernel(x_ptr, o_ptr, N: tl.constexpr):
        i = tl.arange(0, N)[:, None]
        j = tl.arange(0, N)[None, :]
        tile = tl.load(x_ptr + i * N + j)
        # ONE arange, read down both axes. `i - j` is what makes the two reads
        # distinguishable in the output; `i + j` would not be.
        tl.store(o_ptr + i * N + j, tile * (i - j).to(tl.float32))

    return rebuilt_row_read_two_ways_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(N, N)
    i = torch.arange(N).reshape(N, 1)
    j = torch.arange(N).reshape(1, N)
    return {"o": (x * (i - j).to(torch.float32)).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="rebuilt_row_read_two_ways_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N * N,)),
          Arg("o", OUT, "float32", (N * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
)
