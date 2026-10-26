"""Pins an index snapshot with a trip dim: each of T trips gathers rows (r + 3t + pid) mod V of the table,
an index the kernel computes, so the run records one row of the snapshot per trip and the TOG's trip t
reads row t. Breaks if the snapshot takes one row for every trip, or the TOG's gather reads no record."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

S, T, B, D, V = 2, 4, 8, 64, 13


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_gather_computed_index_in_loop_tog(w_ptr, o_ptr, T: tl.constexpr, B: tl.constexpr,
                                                 D: tl.constexpr, V: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, B)
        d = tl.arange(0, D)
        acc = tl.zeros((B, D), dtype=tl.float32)
        for t in range(0, T):
            i = (r + 3 * t + pid) % V
            acc += tl.load(w_ptr + i[:, None] * D + d[None, :])
        tl.store(o_ptr + (pid * B + r)[:, None] * D + d[None, :], acc)

    return rtattr_gather_computed_index_in_loop_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(V, D, generator=g)}


def reference(inputs):
    import torch
    w = inputs["w"]
    return {"o": torch.cat([sum(w[[(r + 3 * t + p) % V for r in range(B)]] for t in range(T)) for p in range(S)])}


SPEC = KernelSpec(
    name="rtattr_gather_computed_index_in_loop_tog", kernel=kernel,
    signature={"w_ptr": "*fp32", "o_ptr": "*fp32", "T": "constexpr", "B": "constexpr", "D": "constexpr",
               "V": "constexpr"},
    constexprs={"T": T, "B": B, "D": D, "V": V},
    args=[Arg("w", IN, "float32", (V, D)), Arg("o", OUT, "float32", (S * B, D))],
    expect_ir={"09-custom.mlir": {"pytorchsim.runtime_attribute": 1}},
    grid=(S,), reference=reference, make_inputs=make_inputs, rtol=1e-5, atol=1e-5,
)
