"""Folding LoRA adapters into a weight before the GEMM --
`W' = W + sum_a scale_a * (B_a @ A_a)`, then `y = x @ W'`.

    dW = 0
    for each adapter a:
        for j in 0 .. r-1:
            dW += scale_a * B_a[:, j] (outer) A_a[j, :]        [K, N] += [K]x[N]
    W' = W + dW
    y  = x @ W'

WHY ANYONE MERGES AT ALL. A LoRA adapter is `W + (alpha/r) B A` with r of 4 to 64
against a K,N of thousands, so `B @ A` is a rank-r correction that costs almost
nothing to STORE and quite a lot to APPLY: keeping it separate means every layer
runs a second matmul, `(x @ B) @ A`, with its own launches and its own read of x.
Merging pays that once and the model then runs at full speed with no adapter in
the graph at all. This is `merge_and_unload` in PEFT, `--lora-merge` in every
diffusion UI, and the fast path in vLLM and TensorRT-LLM when a server is pinned
to one adapter.

  It is also where diffusion and LLM serving want exactly the same kernel.
  A Flux or SDXL checkpoint is routinely run with several style LoRAs at once at
  user-chosen strengths, which is the `sum over a` and the per-adapter `scale`;
  an LLM server merges one adapter per tenant. Same arithmetic, same shape.

  And it is why merging cannot simply be done offline: `scale` is a slider. A
  weight folded at scale 0.8 has to be re-folded when the user drags it to 0.6,
  so this runs at load time, per request, against the live weight.

THE SHAPE THAT NOTHING ELSE IN THE SUITE HAS: A RANK-1 OUTER PRODUCT ACCUMULATED
INTO A RANK-2 TILE. `B[:, j]` is a column vector over the contraction axis and
`A[j, :]` is a row vector over the free axis, and their product covers the whole
[K, N] weight from 2 * (K + N) numbers. Every other rank-2 producer here is a
`tl.dot`; this one is a BROADCAST MULTIPLY of two rank-1 tiles, `b[:, None] *
a[None, :]`, which is the same value the systolic array would compute for a
matmul of inner dimension 1 and a completely different instruction.

  Written as r separate outer products rather than one `tl.dot(B, A)` because
  that is the honest shape: r is 4 here and 4 to 64 in practice, always far below
  the 16 a `tl.dot` needs on its contraction axis, so a real fold either pads the
  rank to the tile size and throws the work away or does exactly this. The two
  rank-1 loads are also the awkward pair -- `B[:, j]` is a STRIDED column read
  (stride r, one element per row) and `A[j, :]` is contiguous -- so one operand
  of every outer product is gathered down the wrong axis.

AND THE RESULT IS THEN A MATMUL OPERAND. `W'` is produced elementwise, tile-wide,
by r * a accumulations, and immediately consumed as the B operand of `tl.dot`.
So a value the vector units built has to land in whatever layout the array
demands, without a round trip -- the constraint swiglu.py applies to matmul
OUTPUTS, applied here to a matmul INPUT. `wf` is stored as well so the fold and
the GEMM can be told apart when they disagree.

TWO ADAPTERS, DIFFERENT SCALES, NON-COMMUTING ONLY IN FLOATING POINT. The sum
over adapters is exact in real arithmetic in any order; in fp32 it is not, and
the loop order here (adapter outer, rank inner) is PEFT's.

NOT MODELLED: the multi-adapter BATCHED case, where different rows of x use
different adapters and the fold cannot happen at all -- that is a segmented GEMM
and a different kernel. This is the single-active-adapter-set merge.

A rank-1 outer product accumulated into a rank-2 tile -- the one rank-2 producer here that is not a tl.dot -- with one operand read down a stride-R column, and the result handed straight to the array as a matmul INPUT. `wf` isolates the fold from the GEMM.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 16                  # tokens
K = 32                  # in features -- the contraction axis
N = 32                  # out features
R = 4                   # LoRA rank; far below the 16 a tl.dot could contract
NA = 2                  # adapters merged at once, each with its own strength


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def fuse_lora_weight_kernel(x_ptr, w_ptr, a_ptr, b_ptr, s_ptr,
                                wf_ptr, y_ptr,
                                M: tl.constexpr, K: tl.constexpr,
                                N: tl.constexpr, R: tl.constexpr,
                                NA: tl.constexpr):
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        rm = tl.arange(0, M)

        wf = tl.load(w_ptr + rk[:, None] * N + rn[None, :])            # [K, N]

        for a in tl.static_range(NA):
            sc = tl.load(s_ptr + a)
            for j in tl.static_range(R):
                # THE OUTER PRODUCT. bj is a STRIDED column of B (stride R),
                # aj is a contiguous row of A; their broadcast product is a
                # full [K, N] tile built from K + N numbers.
                bj = tl.load(b_ptr + a * (K * R) + rk * R + j)          # [K]
                aj = tl.load(a_ptr + a * (R * N) + j * N + rn)          # [N]
                wf += sc * (bj[:, None] * aj[None, :])

        tl.store(wf_ptr + rk[:, None] * N + rn[None, :], wf)

        # The folded weight goes straight into the array as an OPERAND, with no
        # round trip through memory in between.
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], tl.dot(x, wf))

    return fuse_lora_weight_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * K, generator=g, dtype=torch.float32) * 0.5,
            "w": torch.randn(K * N, generator=g, dtype=torch.float32) * 0.2,
            # A is initialised gaussian and B at zero in training; both are
            # non-trivial in a trained checkpoint, which is what gets merged.
            "a": torch.randn(NA * R * N, generator=g, dtype=torch.float32) * 0.3,
            "b": torch.randn(NA * K * R, generator=g, dtype=torch.float32) * 0.3,
            # alpha/r strengths -- the two sliders.
            "scale": torch.tensor([0.8, 0.35], dtype=torch.float32)}


def reference(inputs):
    import torch
    w = inputs["w"].reshape(K, N).clone()
    a = inputs["a"].reshape(NA, R, N)
    b = inputs["b"].reshape(NA, K, R)
    for s in range(NA):
        w = w + inputs["scale"][s] * (b[s] @ a[s])
    y = inputs["x"].reshape(M, K) @ w
    return {"wf": w.reshape(-1).contiguous(), "y": y.reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="fuse_lora_weight_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "w_ptr": "*fp32", "a_ptr": "*fp32",
               "b_ptr": "*fp32", "s_ptr": "*fp32", "wf_ptr": "*fp32",
               "y_ptr": "*fp32", "M": "constexpr", "K": "constexpr",
               "N": "constexpr", "R": "constexpr", "NA": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "R": R, "NA": NA},
    args=[Arg("x", IN, "float32", (M * K,)),
          Arg("w", IN, "float32", (K * N,)),
          Arg("a", IN, "float32", (NA * R * N,)),
          Arg("b", IN, "float32", (NA * K * R,)),
          Arg("scale", IN, "float32", (NA,)),
          Arg("wf", OUT, "float32", (K * N,)),
          Arg("y", OUT, "float32", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
