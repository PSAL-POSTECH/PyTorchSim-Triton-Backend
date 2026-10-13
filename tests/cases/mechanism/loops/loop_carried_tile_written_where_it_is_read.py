"""Two loop-carried tiles, each updated by one fused generic that reads it: the trip writes each
where it reads it, and bufferize leaves no copy of a carried tile behind.
Breaks it: a generic that keeps a fresh tensor.empty as its init, or one that takes a loaded tile.

POSITIVE. `acc` is the LAST input of its fused generic, behind two loaded tiles of its own type, and `run` the FIRST of its own. `expect_ir` holds linalg.generic in the bufferized IR at 2, the two updates. It is 3 on the tree before tritonpytorchsim-reuse-input-as-init (both inits are one shared tensor.empty, and the second update is followed by a whole-tile copy into the carried buffer), and 3 again when the first same-typed input is taken instead of the one the loop hands back (`acc` is then computed into a loaded tile and copied).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
K = 4
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def loop_carried_tile_written_where_it_is_read_kernel(x_ptr, y_ptr, o_ptr, s_ptr,
                                                          K: tl.constexpr,
                                                          BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = x_ptr + r
        q = y_ptr + r
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        run = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in range(0, K):
            a = tl.load(p)
            b = tl.load(q)
            acc = a * b + acc
            run = run - b
            p = p + BLOCK
            q = q + BLOCK
        tl.store(o_ptr + r, acc)
        tl.store(s_ptr + r, run)

    return loop_carried_tile_written_where_it_is_read_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "y": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    x, y = inputs["x"], inputs["y"]
    acc = torch.zeros(BLOCK, dtype=torch.float32)
    run = torch.zeros(BLOCK, dtype=torch.float32)
    for k in range(K):
        a = x[k * BLOCK:(k + 1) * BLOCK]
        b = y[k * BLOCK:(k + 1) * BLOCK]
        acc = a * b + acc
        run = run - b
    return {"o": acc, "s": run}


SPEC = KernelSpec(
    name="loop_carried_tile_written_where_it_is_read_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32", "s_ptr": "*fp32",
               "K": "constexpr", "BLOCK": "constexpr"},
    constexprs={"K": K, "BLOCK": BLOCK},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("y", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (BLOCK,)),
        Arg("s", OUT, "float32", (BLOCK,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    expect_ir={"08-passes/07-torchsim-bufferize.mlir": {"linalg.generic": 2}},
)
