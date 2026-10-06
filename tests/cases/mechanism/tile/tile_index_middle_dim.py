"""`tl.arange` on the MIDDLE axis of a rank-3 tile -- neither lanes nor vector.

WHAT IT PINS. vectorize gives `linalg.index` three answers, one per role
the dim can have in the nest it just built:

    the lane dim      the lane number      torchsim.vlane_idx
    the vector dim    the position in a    vector.step + the loop's offset
                      lane's vector
    anything else     a plain loop of the  broadcast of the induction variable
                      nest

The third one did not exist. The reason given was "a middle dim is a plain loop
counter and has no vector form here", and the second half does not follow from
the first: a loop counter's vector form is the broadcast of it, which is the
same broadcast every captured scalar in that body already gets. So a rank-3
iteration space was unreachable whenever the body asked for its middle index --
and that index is neither hard nor ambiguous, it is constant across the vector
and the same in every lane, because neither the vector nor the lanes run along
that dim.

MEASURED BEFORE THE CHANGE, on bbe05e5:

    adapt: NotVectorisable: linalg.index on dim 1 is neither the lane dim nor
    the vector dim

sort/sort_radix carries the same shape at 128 x 16 x 16 and was refused for the
same reason; this is the fifteen-line form of it.

WHY IT IS A THREE-AXIS TILE AND NOT A TWO-AXIS ONE. With rank 2 every dim IS
either the lane dim or the vector dim, so the missing answer cannot be reached
at all -- which is exactly why nothing noticed for as long as it did.

WHAT WOULD BREAK IT. Deciding what a `linalg.index` means from the axis NUMBER
rather than from the role the nest gave that dim. Which dim rides the lanes is
`torchsim.lane_axis` composed with the maps, and which one the vector runs along is
whichever the nest put innermost; neither is `0` and neither is `rank - 1`.

A linalg.index on the middle dim of a rank-3 nest, which is a plain loop counter and is read as the broadcast of its induction variable. Fails before with 'linalg.index on dim 1 is neither the lane dim nor the vector dim'.

LAYOUT PIN. `expect_ir` holds `torchsim.copy` in 04-passes/04-torchsim-materialize-layouts.mlir at 0; what that pins is the fold treating a plain replica (lanes on a unit axis, one element per lane) as riding its wide axis, not as one bank; held, 1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N = 8, 4, 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_index_middle_dim_kernel(x_ptr, o_ptr, M: tl.constexpr,
                                     K: tl.constexpr, N: tl.constexpr):
        rm = tl.arange(0, M)[:, None, None]
        rk = tl.arange(0, K)[None, :, None]
        rn = tl.arange(0, N)[None, None, :]
        idx = rm * (K * N) + rk * N + rn
        x = tl.load(x_ptr + idx)
        # rk is the MIDDLE axis: not the lane dim (that is rm) and not the
        # vector dim (that is rn). It survives as a linalg.index because its
        # VALUE is used, not just its address.
        tl.store(o_ptr + idx, x + rk.to(tl.float32))

    return tile_index_middle_dim_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, K, N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x = inputs["x"]
    k = torch.arange(K, dtype=torch.float32).reshape(1, K, 1)
    return {"o": x + k}


SPEC = KernelSpec(
    name="tile_index_middle_dim_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "M": "constexpr",
               "K": "constexpr", "N": "constexpr"},
    constexprs={"M": M, "K": K, "N": N},
    args=[Arg("x", IN, "float32", (M, K, N)),
          Arg("o", OUT, "float32", (M, K, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"04-passes/04-torchsim-materialize-layouts.mlir": {"torchsim.copy": 0}},
)
