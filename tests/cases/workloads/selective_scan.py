"""Mamba's selective scan -- a first-order recurrence with input-dependent gates.

    h_t = a_t * h_{t-1} + b_t          a and b depend on x_t
    y_t = c_t * h_t

A SCAN THAT IS NOT A CUMSUM. `sampling.py` has the suite's only other scan and
it is `cumsum`: the combining operation is addition, which is associative with
no state beyond the running total. This one carries a MULTIPLIER as well, and
the multiplier is computed from the input -- that is what "selective" means, and
it is the whole difference between Mamba and the linear state-space models
before it.

  Why selectivity matters: a fixed `a` makes the model forget at a fixed rate,
  which is a convolution and cannot decide that one token deserves remembering.
  An input-dependent `a` can hold a value across a thousand steps and drop it on
  the next. That is the property attention has and earlier SSMs did not.

IT IS STILL ASSOCIATIVE, AND THAT IS THE ONLY REASON IT IS PARALLEL AT ALL.
The pair `(a, b)` composes as

    (a1, b1) . (a2, b2) = (a1 a2, a2 b1 + b2)

so a Blelloch scan computes all prefixes in log(T) depth. Mamba's speed rests
entirely on that identity -- written as a sequential loop it is one multiply-add
per step with no parallelism at all.

WHAT IT ASKS OF THIS MACHINE, AND WHY IT IS WRITTEN SEQUENTIALLY HERE. There is
no scan instruction: `_unroll_scan` UNROLLS `tl.cumsum`,
which is fine at 64 and impossible at 4096. A gated scan has no `tl.` form at
all, so it has to be a loop over T carrying `h` as an `scf.for` iteration
argument -- which is the shape `dl/attention` established, one state per lane
rather than a tile.

  So this passes and it does not mean Mamba runs. The loop is sequential in T,
  and T is the sequence length: the parallel form needs the associative scan
  above, and building that would mean log(T) stages of a lane-crossing
  shuffle -- which this machine does not have. Recorded rather than attempted.

torch equivalent:
    h = 0
    for t in range(T):  h = a[t] * h + b[t];  y[t] = c[t] * h

Mamba: a gated scan, sequential in T with the state as an scf.for iteration argument. The PARALLEL form needs a lane-crossing associative scan this machine has no instruction for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, N = 32, 64             # steps, and independent channels scanned in parallel


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def selective_scan_kernel(a_ptr, b_ptr, c_ptr, o_ptr,
                              T: tl.constexpr, N: tl.constexpr):
        n = tl.arange(0, N)
        # THE STATE IS ONE VALUE PER CHANNEL, carried across the loop. Channels
        # are independent, so the parallelism is over N and the loop is over T.
        h = tl.zeros((N,), dtype=tl.float32)
        for t in range(T):
            a = tl.load(a_ptr + t * N + n)
            b = tl.load(b_ptr + t * N + n)
            c = tl.load(c_ptr + t * N + n)
            h = a * h + b
            tl.store(o_ptr + t * N + n, c * h)

    return selective_scan_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.rand(T * N, generator=g, dtype=torch.float32) * 0.9,
            "b": torch.randn(T * N, generator=g, dtype=torch.float32),
            "c": torch.randn(T * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"].reshape(T, N)
    b = inputs["b"].reshape(T, N)
    c = inputs["c"].reshape(T, N)
    h = torch.zeros(N)
    out = torch.empty(T, N)
    for t in range(T):
        h = a[t] * h + b[t]
        out[t] = c[t] * h
    return {"o": out.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="selective_scan_kernel",
    kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "o_ptr": "*fp32", "T": "constexpr", "N": "constexpr"},
    constexprs={"T": T, "N": N},
    args=[Arg("a", IN, "float32", (T * N,)),
          Arg("b", IN, "float32", (T * N,)),
          Arg("c", IN, "float32", (T * N,)),
          Arg("o", OUT, "float32", (T * N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
