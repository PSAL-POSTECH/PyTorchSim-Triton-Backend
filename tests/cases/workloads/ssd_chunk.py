"""Mamba-2's chunked SSD -- a quadratic form inside the chunk, a recurrence between.

Per step the model is still the linear recurrence `selective_scan.py` runs:

    h_t = a_t * h_{t-1} + B_t * x_t          h in R^N, a_t and x_t scalars
    y_t = C_t . h_t

WHY IT IS NOT COMPUTED THAT WAY. Expanding the recurrence gives

    y_t = sum_{s<=t}  exp(cum[t] - cum[s]) * (C_t . B_s) * x_s     cum = cumsum(log a)

which is a causally-masked, decay-weighted attention matrix -- the "duality" in
State Space Duality. The recurrent form is O(T*N) work with a serial dependency
T long; the quadratic form is O(T^2*N) work with NO dependency at all and lands
on a matmul unit. Mamba-2 takes neither: it cuts the sequence into chunks of Q,
uses the quadratic form INSIDE a chunk (where Q^2 is affordable and the whole
chunk is one dot) and the recurrent form BETWEEN chunks (where the state is N
numbers and there are only T/Q steps). That single choice is why Mamba-2 trains
2-8x faster than Mamba-1 at the same quality.

WHAT IS DIFFERENT FROM selective_scan.py, WHICH IS THE SAME MATH. That kernel is
the pure recurrence: one multiply-add per timestep, state as an `scf.for`
iteration argument, T iterations, no matmul. Here the loop is T/Q iterations
long instead of T, and each iteration contains

  * a `tl.dot` producing a [Q, Q] tile -- `C @ B^T`, the whole chunk's
    interactions at once. Nothing in selective_scan.py is a matmul;
  * a causal mask and a DECAY matrix over that tile, `exp(cum[t] - cum[s])`,
    which is a rank-1 difference broadcast both ways across a square tile;
  * the state carry, which is now the only sequential part.

THE SEGMENT SUM IS WRITTEN AS A MASKED ROW SUM, and that is deliberate. `cum` is
a prefix sum along the chunk, and a prefix sum along the LANE axis has no
instruction here. So it is computed as `sum_s (s <= t) log a[s]` over a [Q, Q]
tile -- reduction on the inner axis, which is the only direction that folds. It
is also exactly the `segsum` helper in Mamba-2's own reference implementation,
for the same reason at a different scale: the materialised form is numerically
better behaved than a long product of decays.

B IS READ TRANSPOSED, AND THAT IS FORCED. The new state is `sum_t (...) * B[t]`
-- a reduction over TIME, and time is the lane axis for every other tile in the
chunk. Reading B as [N, Q] instead of [Q, N] puts the reduced axis in-lane, at
the cost of a strided load (`lm_head_tied.py` pays the same toll for the same
reason). The alternative is a fold across the banks, which does not exist.

BOUNDS: `a` is in (0, 1), so `cum` is non-positive and every exponential in here
is a decay in [0, 1]. That is not an accident of the test data -- it is why the
chunked form is stable where a running product of gains would not be.

Mamba-2: a [Q,Q] dot with a decay mask inside each chunk and an scf.for carrying the state between them. The chunk states are stored so the boundary can be checked apart from the interior.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

Q, N, NC = 16, 16, 4            # chunk length, state size, chunk count
L = Q * NC                      # sequence length


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ssd_chunk_kernel(a_ptr, x_ptr, b_ptr, c_ptr, y_ptr, h_ptr,
                         Q: tl.constexpr, N: tl.constexpr, NC: tl.constexpr):
        t = tl.arange(0, Q)[:, None]            # output step, within the chunk
        s = tl.arange(0, Q)[None, :]            # input step, within the chunk
        n = tl.arange(0, N)

        h = tl.zeros((N,), dtype=tl.float32)    # the state carried BETWEEN chunks
        for ck in tl.static_range(NC):
            base = ck * Q
            la = tl.log(tl.load(a_ptr + base + s))              # [1, Q]

            # SEGMENT SUM as a masked row sum -- a prefix scan folded onto the
            # inner axis, because the lane axis cannot fold.
            cum = tl.sum(tl.where(s <= t, la, 0.0), axis=1)     # [Q]
            tot = tl.sum(la, axis=1)                            # (1,) whole chunk

            xs = tl.load(x_ptr + base + s)                      # [1, Q]
            Cq = tl.load(c_ptr + (base + t) * N + n[None, :])   # [Q, N]
            Bt = tl.load(b_ptr + (base + s) * N + n[:, None])   # [N, Q] strided

            # INSIDE THE CHUNK: the quadratic form. One dot for Q^2 interactions.
            G = tl.dot(Cq, Bt)                                  # [Q, Q]
            D = tl.where(s <= t, tl.exp(cum[:, None] - cum[None, :]), 0.0)
            y = tl.sum(D * G * xs, axis=1)                      # [Q]

            # ACROSS THE BOUNDARY: what the previous chunk left, decayed to here.
            y += tl.exp(cum) * tl.sum(Cq * h[None, :], axis=1)
            tl.store(y_ptr + base + tl.arange(0, Q), y)

            # THE NEW STATE: a reduction over time, which is why Bt is [N, Q].
            h = tl.exp(tot) * h + tl.sum(Bt * (tl.exp(tot - cum)[None, :] * xs),
                                         axis=1)
            tl.store(h_ptr + ck * N + n, h)

    return ssd_chunk_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.rand(L, generator=g, dtype=torch.float32) * 0.3 + 0.7,
            "x": torch.randn(L, generator=g, dtype=torch.float32),
            "b": torch.randn(L * N, generator=g, dtype=torch.float32) * 0.5,
            "c": torch.randn(L * N, generator=g, dtype=torch.float32) * 0.5}


def reference(inputs):
    import torch
    a = inputs["a"]
    x = inputs["x"]
    B = inputs["b"].reshape(L, N)
    C = inputs["c"].reshape(L, N)
    h = torch.zeros(N, dtype=torch.float32)
    y = torch.empty(L, dtype=torch.float32)
    hs = torch.empty(NC, N, dtype=torch.float32)
    for i in range(L):
        h = a[i] * h + B[i] * x[i]
        y[i] = (C[i] * h).sum()
        if (i + 1) % Q == 0:
            hs[i // Q] = h
    return {"y": y.contiguous(), "h": hs.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="ssd_chunk_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "x_ptr": "*fp32", "b_ptr": "*fp32",
               "c_ptr": "*fp32", "y_ptr": "*fp32", "h_ptr": "*fp32",
               "Q": "constexpr", "N": "constexpr", "NC": "constexpr"},
    constexprs={"Q": Q, "N": N, "NC": NC},
    args=[Arg("a", IN, "float32", (L,)),
          Arg("x", IN, "float32", (L,)),
          Arg("b", IN, "float32", (L * N,)),
          Arg("c", IN, "float32", (L * N,)),
          Arg("y", OUT, "float32", (L,)),
          Arg("h", OUT, "float32", (NC * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-3, atol=1e-3,
    expect="blocked",
    expect_why="matmul K=16 is not whole lane tiles of 128; the matrix unit's lowering masks no lanes yet (#81)",
)
