"""A row on the lanes swaps each element with the one 2^k lanes over: the butterfly a bitonic step
reads its partner through.

WHAT IT PINS. `x.reshape(-1, 2, s).flip(1)` is lane `l` reading lane `l ^ s`. No one run of lanes
carries that map (the source steps +s, then -s, every s lanes), so the permute is a MAPPED round:
one pattern, `lane ^ s`, set once with xlu_set_pattern, and ONE push of the row, as the v6e sets
`vlaneseq ^ s` with vset.pattern.permute before one vperm a register. Before, each run of lanes was
its own round, 64 pushes of the same row for s = 1.

The row is computed (exp) before it is reordered, so it is already on the lanes: a load alone would
be reordered by the DMA instead.

torch equivalent: x.exp().reshape(-1, 2, s).flip(1).reshape(-1), for s = 1 and s = 16.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_xlu_permute_lane_xor_kernel(x_ptr, a_ptr, b_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.exp(tl.load(x_ptr + r))
        a = tl.reshape(tl.flip(tl.reshape(x, (N // 2, 2, 1)), 1), (N,))
        b = tl.reshape(tl.flip(tl.reshape(x, (N // 32, 2, 16)), 1), (N,))
        tl.store(a_ptr + r, a)
        tl.store(b_ptr + r, b)

    return ops_xlu_permute_lane_xor_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    x = inputs["x"].exp()
    return {"a": x.reshape(-1, 2, 1).flip(1).reshape(-1), "b": x.reshape(-1, 2, 16).flip(1).reshape(-1)}


SPEC = KernelSpec(
    name="ops_xlu_permute_lane_xor_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("a", OUT, "float32", (N,)), Arg("b", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
