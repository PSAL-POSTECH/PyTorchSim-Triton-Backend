"""Pins a gather under a branch inside a loop as a refusal by name: trips whose flag is set gather rows by
the index they read, the others none, so the snapshot's rows would not be one per trip. Breaks if it is
recorded, read without its record, or the kernel stops anywhere else."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT
from pytorchsim_triton_compiler.contract.stop import Reason, Stop

T, B, D, V = 4, 8, 64, 32
IDS = [(7 * k + 1) % V for k in range(T * B)]
FLAGS = [1, 0, 1, 1]


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rtattr_gather_under_branch_in_loop_tog(w_ptr, ids_ptr, f_ptr, o_ptr, T: tl.constexpr,
                                               B: tl.constexpr, D: tl.constexpr):
        r = tl.arange(0, B)
        d = tl.arange(0, D)
        for t in range(0, T):
            if tl.load(f_ptr + t) != 0:
                ids = tl.load(ids_ptr + t * B + r)
                tl.store(o_ptr + (t * B + r)[:, None] * D + d[None, :],
                         tl.load(w_ptr + ids[:, None] * D + d[None, :]))

    return rtattr_gather_under_branch_in_loop_tog


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(V, D, generator=g), "ids": torch.tensor(IDS, dtype=torch.int32),
            "f": torch.tensor(FLAGS, dtype=torch.int32), "o": torch.zeros(T * B, D)}


def reference(inputs):
    o = inputs["o"].clone()
    for t, f in enumerate(FLAGS):
        if f:
            o[t * B:(t + 1) * B] = inputs["w"][IDS[t * B:(t + 1) * B]]
    return {"o": o}


SPEC = KernelSpec(
    name="rtattr_gather_under_branch_in_loop_tog", kernel=kernel,
    signature={"w_ptr": "*fp32", "ids_ptr": "*i32", "f_ptr": "*i32", "o_ptr": "*fp32", "T": "constexpr",
               "B": "constexpr", "D": "constexpr"},
    constexprs={"T": T, "B": B, "D": D},
    args=[Arg("w", IN, "float32", (V, D)), Arg("ids", IN, "int32", (T * B,)), Arg("f", IN, "int32", (T,)),
          Arg("o", OUT, "float32", (T * B, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs, rtol=1e-6, atol=1e-6,
    expect="refuse", expect_stop=Stop("lower", Reason.RUNTIME_ATTRIBUTE),
    expect_why="the gather sits under `if f[t]` inside the trip loop: its snapshot would hold rows for "
               "the trips that ran alone, not one per trip",
)
