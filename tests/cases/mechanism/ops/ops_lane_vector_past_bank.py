"""A tile wider ACROSS the lanes than it is tall, and the vector that ran off it.

WHAT IT PINS. vectorize has two loop nests for a rank-2 tile and picks
between them with `covered`: the dim-0 nest is allowed to stand in for a
last-axis-banked tile when its one vector reaches every element a lane holds.
That question was asked and a second one was not -- whether the vector STAYS
INSIDE the lane's bank.

    [M, N] banked on axis 1, on `lanes` lanes    a lane holds M elements
    the dim-0 nest issues                        vector<N>

With N > M those are different numbers and the surplus is written into the NEXT
memref.global. The reservation sizes each bank by what the tile needs -- it
has no idea a wider vector is coming -- and its own
docstring says why that matters: "Those two disagreeing writes past the end of
a global". This was the second reader spelling the formula itself.

WHAT MAKES IT VISIBLE HERE. An overrun only shows when the global it lands in is
still LIVE, so this kernel keeps one:

    `x` is loop-invariant     loaded once, read by every iteration
    the loop stores over it   the [M, N] result's bank is followed by x's

So iteration 0 is right, its store clobbers `x`, and every later iteration adds
its row to whatever was left there. Measured on dl/rnnt_joint, a [16, 32] tile
on 32 lanes: u=0 exact and u=1..3 equal to `tanh(W[h, t] + g[0][h]) + g[u][h]`
to 2.4e-07 against spike memory -- the previous iteration's tanh, read where `f`
should have been. That kernel COMPILED and VERIFIED for as long as the buffer it
clobbered happened to be dead, which is why the reproducer is here and not left
inside it.

WHY THE MATMUL. The lane axis has to be the LAST one, and the matrix unit is
the only thing that demands it -- an elementwise tile banks on dim 0. `tl.dot`
against a [N, V] operand is the smallest way to say so. The rank-2 shapes are
otherwise unremarkable: what the file is about is M < N.

WHAT WOULD BREAK IT. Ask `shape[-1] >= shape[0]` and nothing else,
which is coverage without containment. The kernel then builds, runs, and returns
the wrong numbers with no diagnostic anywhere -- the failure mode this suite
exists to make loud.

torch equivalent: for i in range(U): o[i] = (x + r[i]) @ w.T

A last-axis-banked tile whose lane extent exceeds its in-lane one. The nest's vector must both cover the lane's elements and stay inside its reservation; asking only the first writes into the next global, and only a live neighbour makes that visible.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, V, U = 16, 32, 32, 3     # M < N: a lane holds 16 and the vector wants 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_lane_vector_past_bank_kernel(x_ptr, r_ptr, w_ptr, o_ptr,
                                         M: tl.constexpr, N: tl.constexpr,
                                         V: tl.constexpr, U: tl.constexpr):
        m = tl.arange(0, M)[:, None]
        n = tl.arange(0, N)[None, :]
        kt = tl.arange(0, N)[:, None]
        v = tl.arange(0, V)[None, :]
        # Loop invariant, and the buffer the overrun lands in.
        x = tl.load(x_ptr + m * N + n)                    # [M, N]
        # Read transposed, so the array's operand is [N, V] and the lane axis
        # of everything it touches is the LAST one.
        wt = tl.load(w_ptr + v * N + kt)                  # [N, V]

        for i in tl.static_range(U):
            r = tl.load(r_ptr + i * N + tl.arange(0, N))  # a row per iteration
            z = x + r[None, :]                            # [M, N], banked on 1
            tl.store(o_ptr + i * (M * V) + m * V + v, tl.dot(z, wt))

    return ops_lane_vector_past_bank_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)
    return {"x": r(M * N) * 0.5, "r": r(U * N) * 0.5, "w": r(V * N) * (N ** -0.5)}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(M, N)
    rr = inputs["r"].reshape(U, N)
    w = inputs["w"].reshape(V, N)
    o = torch.stack([(x + rr[i]) @ w.T for i in range(U)])
    return {"o": o.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="ops_lane_vector_past_bank_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "r_ptr": "*fp32", "w_ptr": "*fp32",
               "o_ptr": "*fp32", "M": "constexpr", "N": "constexpr",
               "V": "constexpr", "U": "constexpr"},
    constexprs={"M": M, "N": N, "V": V, "U": U},
    args=[Arg("x", IN, "float32", (M * N,)), Arg("r", IN, "float32", (U * N,)),
          Arg("w", IN, "float32", (V * N,)),
          Arg("o", OUT, "float32", (U * M * V,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
