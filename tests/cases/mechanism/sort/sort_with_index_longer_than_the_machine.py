"""Inductor's sort_with_index over four lane-fulls: 1024 i64 values and their indices on 256 lanes,
the shape qwen's MoE routing sort takes at 32 lanes (128 wide) and deepseek's at 8. Two things had
to give. The bitonic stages reshape the pair to [.., 2, 2^stage], and at one stage TritonGPU lays
a dim across the lanes and a register at once, which no physical order holds: materialize_layouts
now lays it as the nearest layout the machine has. Then the sort's several hundred i64 tiles, each
with its own scratchpad global, claimed 309 KB a lane against a budget of 256 KB: bufferize now
shares a global among the tiles whose lives do not overlap, 121 KB. Breaks if a re-laid tile keeps
the dim split (no plan), or if every alloc is a global of its own again (the budget).

A values-and-indices sort over four lane-fulls, through the layout the machine has and the
scratchpad its live tiles need.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024


def kernel():
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers

    @triton.jit
    def sort_with_index_longer_than_the_machine_kernel(x_ptr, v_ptr, i_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.load(x_ptr + r)
        v, i = triton_helpers.sort_with_index(x, r.to(tl.int64), None, 0, stable=False,
                                              descending=False)
        tl.store(v_ptr + r, v)
        tl.store(i_ptr + r, i)

    return sort_with_index_longer_than_the_machine_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randperm(N, generator=g).to(torch.int64)}


def reference(inputs):
    v, i = inputs["x"].sort(stable=True)
    return {"v": v, "i": i}


SPEC = KernelSpec(
    name="sort_with_index_longer_than_the_machine_kernel", kernel=kernel,
    signature={"x_ptr": "*i64", "v_ptr": "*i64", "i_ptr": "*i64", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "int64", (N,)), Arg("v", OUT, "int64", (N,)), Arg("i", OUT, "int64", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
