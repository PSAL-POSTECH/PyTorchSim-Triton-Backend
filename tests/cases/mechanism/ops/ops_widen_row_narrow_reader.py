"""A row a broadcast wants WIDE and another generic still reads NARROW.

WHAT IT PINS, AND IT IS NO LONGER A PASS'S READER CHECK. widen_broadcast_loads
turned a `[1, N]` load that a broadcast reads into an `[M, N]` one by rewriting
the LOAD'S TYPE, and a type is a statement to EVERYONE that touches it (rule 4):
`rb * ra` iterates ONE row, so writing the identity onto its `ra` slot states an
iteration extent of M that its own outs -- still `[1, N]` -- contradicts. That
pass enumerated the readers and declined the whole group, and this file was the
smallest shape that reached the decline.

    'linalg.generic' op inferred input/output operand #1 has shape's dimension
    #0 to be 1, but found 16

THE PASS IS GONE and the shape is kept, because what it now pins is the reason
the deletion was safe. fit_layout_to_lanes._clone_loaded_row_wide does the same
staging from the other end: it MINTS a second buffer, gives it a zero DRAM
stride on the broadcast axis and re-points ONE operand. Nothing is retyped, so
"can every reader be told" is not a question it has to ask -- the narrow reader
keeps the narrow row and never learns there is a wide one. A regression here
would be a staging that retypes again, and the verifier message above is what it
would say.

WHY THE ROWS ARE LOADED 2-D. Kept from when the pass only widened a load whose
descriptor already carried a zero leading stride and declined a rank-1 row that
anything but the `expand_shape` read. It is still the shape that puts two
readers of different ranks on one loaded row, which is the whole point.

torch equivalent: (x + a, b * a).

A row is widened only when EVERY reader can be told; a reader still iterating one row cannot, so the widening is declined rather than done to half the group.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N = 16, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_widen_row_narrow_reader_kernel(x_ptr, a_ptr, b_ptr, o_ptr, n_ptr,
                                           M: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        # [1, N] tiles whose leading stride is 0 -- one row, described as
        # repeated, which is what makes a widening free when it is available.
        ra = tl.load(a_ptr + rn[None, :])                        # (1, N)
        rb = tl.load(b_ptr + rn[None, :])                        # (1, N)
        na = rb * ra              # `ra` as an INS of an op iterating ONE row
        tile = tl.load(x_ptr + rm[:, None] * N + rn[None, :])    # (M, N)
        out = tile + ra           # ... and as the source of a broadcast
        tl.store(o_ptr + rm[:, None] * N + rn[None, :], out)
        tl.store(n_ptr + rn[None, :], na)

    return ops_widen_row_narrow_reader_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32),
            "a": torch.randn(N, generator=g, dtype=torch.float32),
            "b": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].reshape(M, N)
    a = inputs["a"].reshape(1, N)
    b = inputs["b"].reshape(1, N)
    return {"o": (x + a).reshape(-1), "n": (b * a).reshape(-1)}


SPEC = KernelSpec(
    name="ops_widen_row_narrow_reader_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32",
               "o_ptr": "*fp32", "n_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "N": N},
    args=[Arg("x", IN, "float32", (M * N,)), Arg("a", IN, "float32", (N,)),
          Arg("b", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (M * N,)), Arg("n", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
