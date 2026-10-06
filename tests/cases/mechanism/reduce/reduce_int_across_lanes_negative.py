"""An i32 sum ACROSS the lanes, with negative summands.

THE CROSSBAR DOES NOT COMBINE THIS ONE AND DOES NOT HAVE TO. Its queues hold
`float` and an e32 push REINTERPRETS the lane, so an integer folded there is the
float its bits spell -- 0xFFFFFFFF is a NaN. The answer is not an integer adder
in the unit: a cross-lane unit MOVES 32 bits and the arithmetic belongs to the
vector unit, which is the half that knows the type. So the fold is a transpose
and then the lane's own reduction, which is the shape a TPU lowers every
cross-lane reduction to (XLU `vxpose.b32`, then a `vadd` tree in the VPU).

    non-negative and small   the reinterpretation is a DENORMAL, so float
                             addition adds the bit patterns exactly and the wrong
                             path answered right -- which is why
                             workloads/moe_align_block_size.py, three i32 folds
                             of counts, passed at 0 while this was broken.
    negative                 what this kernel is for.

torch equivalent: x.sum(dim=1) and x.sum(dim=0), int32.

pins the transpose-then-fold path: the crossbar has no integer add and needs none. Both folds of one tile, so one of them rides the lanes whichever axis is banked.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

# The shape moe_align_block_size's histogram has: the WIDER axis takes the lanes,
# so folding it is the crossbar's and not the lane's own.
E, T = 4, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_int_across_lanes_negative_kernel(x_ptr, o_ptr, q_ptr,
                                                E: tl.constexpr,
                                                T: tl.constexpr):
        re = tl.arange(0, E)
        rt = tl.arange(0, T)
        x = tl.load(x_ptr + re[:, None] * T + rt[None, :])
        # BOTH FOLDS OF ONE TILE, which is what moe_align_block_size has: one of
        # them is the lane axis whichever axis layout assignment banks on, and the wider one
        # wins, so the per-expert sum is the crossbar's.
        tl.store(o_ptr + re, tl.sum(x, axis=1))
        tl.store(q_ptr + rt, tl.sum(x, axis=0))

    return reduce_int_across_lanes_negative_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randint(-50, 50, (E, T), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    return {"o": inputs["x"].sum(dim=1, dtype=inputs["x"].dtype),
            "q": inputs["x"].sum(dim=0, dtype=inputs["x"].dtype)}


SPEC = KernelSpec(
    name="reduce_int_across_lanes_negative_kernel", kernel=kernel,
    signature={"x_ptr": "*i32", "o_ptr": "*i32", "q_ptr": "*i32",
               "E": "constexpr", "T": "constexpr"},
    constexprs={"E": E, "T": T},
    args=[Arg("x", IN, "int32", (E, T)), Arg("o", OUT, "int32", (E,)),
          Arg("q", OUT, "int32", (T,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
