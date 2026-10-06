"""A cross-lane `max` over a row that is entirely -inf -- the fully masked row.

THE IDENTITY IS SPELLED TWICE AND THE TWO SPELLINGS DIFFER. lower_to_vcix pads the
lanes a tile does not reach with the reduction's own identity, `-inf`
(`_identity_attr`). spike's crossbar seeded its accumulator at `-FLT_MAX`
instead (`cross_lane_unit.cc reduce`), and `-inf < -FLT_MAX`, so the unit can
never answer `-inf` however many of its inputs are.

Attention is where this row comes from: a query position whose whole key row is
masked leaves softmax's running max at its seed. torch answers `-inf`.

torch equivalent: x.amax(dim=...) where one slice is all -inf.

the identity of a reduction is written in lower_to_vcix and again in spike's reduce, and the two do not agree on the value.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# Both folds of one tile, so the wider axis takes the lanes and one fold is the
# crossbar's -- the shape moe_align_block_size has.
E, T = 4, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_masked_row_all_neg_inf_kernel(x_ptr, o_ptr, q_ptr,
                                             E: tl.constexpr, T: tl.constexpr):
        re = tl.arange(0, E)
        rt = tl.arange(0, T)
        x = tl.load(x_ptr + re[:, None] * T + rt[None, :])
        tl.store(o_ptr + re, tl.max(x, axis=1))
        tl.store(q_ptr + rt, tl.max(x, axis=0))

    return reduce_masked_row_all_neg_inf_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(E, T, generator=g, dtype=torch.float32)
    # COLUMN 5 is the fully masked one, because the fold that crosses the lanes
    # is the one over E -- a masked ROW is folded inside a lane and never reaches
    # the crossbar.
    x[:, 5] = float("-inf")
    return {"x": x}


def reference(inputs):
    return {"o": inputs["x"].amax(dim=1), "q": inputs["x"].amax(dim=0)}


SPEC = KernelSpec(
    name="reduce_masked_row_all_neg_inf_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "q_ptr": "*fp32",
               "E": "constexpr", "T": "constexpr"},
    constexprs={"E": E, "T": T},
    args=[Arg("x", IN, "float32", (E, T)), Arg("o", OUT, "float32", (E,)),
          Arg("q", OUT, "float32", (T,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
