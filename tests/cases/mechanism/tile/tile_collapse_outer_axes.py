"""A rank-3 tile collapsed on its OUTER axes, then fed to the array.

    z = maximum(x, y)            4 x 4 x 16, computed in the scratchpad
    m = tl.reshape(z, (16, 16))  [[0,1],[2]] -- the last axis is untouched
    acc = tl.dot(m, b)

WHY IT IS FREE, and what used to pay for it. A reassociation moves no element
between LANES while the axis the lanes carry is alone in its group (rule 6a), so
`m` and `z` are one image read two ways and fit_layout_to_lanes owes nothing.
`_splits_the_outer` asked instead whether GROUP 0 held two real axes -- true of
every collapse of outer axes -- and minted a DRAM round trip for it. That test
was written for `(16,) -> (2, 8)`, where group 0 IS the lane axis's group; at
rank 3 it names the axes furthest from the lanes.

THE OTHER HALF IS THE BANK ORDER, and it is pinned by the number rather than by
a count. `tile_strides` laid the non-lane axes out axis-0-FASTEST for a last-axis
lane and row-major everywhere else, so `4x4x16` on lane 2 banked (i + 4j) where
the collapse says (4i + j) -- the same bytes in a different order, which is a
wrong answer and not a failure. Reverting it alone gives 14.04 here.

WHAT THIS ONE DOES NOT WITNESS. lower_to_vcix decides an operand was staged by
walking the store's buffer to it, and two SIBLING views of one allocation reach
each other in neither direction. Here bufferize gives the pool's output the
buffer a DMA already filled, so the transfer answers instead;
tile_matmul_reads_a_sibling_view is the kernel that keeps it apart.

A collapse of the outer axes leaves the lane axis alone, so it is a renumbering and not a round trip.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

I, J, K, N = 4, 4, 16, 16
#: The outer two axes stride by 256 and 32, and 256 != 4 * 32 -- so the collapse
#: is NOT linear in the DRAM walk and absorb_layout_ops refuses it, which is what
#: leaves it standing for fit_layout_to_lanes to answer. A contiguous window
#: (64, 16) is absorbed into the descriptor and never reaches this question.
SI, SJ = 256, 32
ASIZE = 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def collapse_outer_axes(a_ptr, b_ptr, o_ptr, I: tl.constexpr,
                            J: tl.constexpr, K: tl.constexpr,
                            N: tl.constexpr, SI: tl.constexpr,
                            SJ: tl.constexpr):
        off = (tl.arange(0, I)[:, None, None] * SI
               + tl.arange(0, J)[None, :, None] * SJ
               + tl.arange(0, K)[None, None, :])
        x = tl.load(a_ptr + off)
        y = tl.load(a_ptr + off + K)
        z = tl.maximum(x, y)
        m = tl.reshape(z, (I * J, K))
        b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N
                    + tl.arange(0, N)[None, :])
        acc = tl.dot(m, b)
        tl.store(o_ptr + tl.arange(0, I * J)[:, None] * N
                 + tl.arange(0, N)[None, :], acc)

    return collapse_outer_axes


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(ASIZE, generator=g, dtype=torch.float32),
            "b": torch.randn(K * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    a, b = inputs["a"], inputs["b"]
    import torch
    idx = (torch.arange(I)[:, None, None] * SI
           + torch.arange(J)[None, :, None] * SJ
           + torch.arange(K)[None, None, :])
    z = a[idx].maximum(a[idx + K])
    return {"o": (z.reshape(I * J, K) @ b.reshape(K, N)).reshape(-1)
            .contiguous()}


SPEC = KernelSpec(
    name="collapse_outer_axes", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "I": "constexpr", "J": "constexpr", "K": "constexpr",
               "N": "constexpr", "SI": "constexpr", "SJ": "constexpr"},
    constexprs={"I": I, "J": J, "K": K, "N": N, "SI": SI, "SJ": SJ},
    args=[Arg("a", IN, "float32", (ASIZE,)),
          Arg("b", IN, "float32", (K * N,)),
          Arg("o", OUT, "float32", (I * J * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    expect_ir={"08-transformed.mlir": {"pytorchsim.crosslane_move": 0}},
)
