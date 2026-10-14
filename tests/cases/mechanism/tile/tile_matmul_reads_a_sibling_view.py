"""A matmul operand and the store that filled it are SIBLING views of one buffer.

    z    = max of four taps          4 x 4 x 16, its own scratchpad buffer
    m    = tl.reshape(z, (16, 16))   a collapse view of that buffer
    out  = dot(dot(m, kt), v) + m    m is read twice, so it is not fused away

WHAT IT PINS. lower_to_vcix decides an operand was STAGED by walking from a
preceding store or transfer to it. Here vectorize writes the pool through a
per-lane `memref.reinterpret_cast` of the buffer while the matmul reads a
`memref.collapse_shape` of the same buffer -- two views of one allocation,
neither of which reaches the other. `_reaches` in either direction answers no,
`isAInit` stays false, and the matmul falls through the gemm lowering to be
refused a pass later as "declared these would not survive it".

WHY EVERY PIECE IS LOAD-BEARING, measured by removing each:

    the outer axes stride 256/32   a contiguous window merges into the
                                   descriptor and absorb_layout_ops takes the
                                   collapse away (tile_collapse_outer_axes)
    FOUR taps, not two             with two, bufferize gives the pool the
                                   buffer a DMA already filled, and the
                                   transfer arm answers instead of the store
    `+ m`, the second use          with one use the pool FUSES into the dot,
                                   which turns the named matmul into a generic,
                                   and absorb_layout_ops then folds the collapse
                                   into its maps -- no view left to mistake
    TWO dots, the second writing   one reader lets absorb_layout_ops take the
    16x16                          collapse forward into the matmul

That is sam2_qpool_attn's shape with the softmax taken out; it was the only
kernel in the suite carrying it, which is why this one exists.

The matmul's A is a collapse view of the pool's buffer while the store that filled it went through a per-lane reinterpret_cast of the same buffer -- siblings, reaching each other in neither direction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

I, J, K, N = 4, 4, 16, 64
M = I * J
SI, SJ = 256, 32
ASIZE = 1024


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def matmul_reads_a_sibling_view(
            a_ptr, b_ptr, c_ptr, o_ptr, I: tl.constexpr, J: tl.constexpr,
            K: tl.constexpr, N: tl.constexpr, M: tl.constexpr,
            SI: tl.constexpr, SJ: tl.constexpr):
        off = (tl.arange(0, I)[:, None, None] * SI
               + tl.arange(0, J)[None, :, None] * SJ
               + tl.arange(0, K)[None, None, :])
        z = tl.maximum(tl.maximum(tl.load(a_ptr + off),
                                  tl.load(a_ptr + off + K)),
                       tl.maximum(tl.load(a_ptr + off + 2 * K),
                                  tl.load(a_ptr + off + 3 * K)))
        m = tl.reshape(z, (M, K))
        kt = tl.load(b_ptr + tl.arange(0, K)[:, None] * N
                     + tl.arange(0, N)[None, :])
        s = tl.dot(m, kt) * 0.25
        v = tl.load(c_ptr + tl.arange(0, N)[:, None] * K
                    + tl.arange(0, K)[None, :])
        out = tl.dot(s, v) + m
        tl.store(o_ptr + tl.arange(0, M)[:, None] * K
                 + tl.arange(0, K)[None, :], out)

    return matmul_reads_a_sibling_view


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(ASIZE, generator=g, dtype=torch.float32),
            "b": torch.randn(K * N, generator=g, dtype=torch.float32),
            "c": torch.randn(N * K, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    a = inputs["a"]
    idx = (torch.arange(I)[:, None, None] * SI
           + torch.arange(J)[None, :, None] * SJ
           + torch.arange(K)[None, None, :])
    z = a[idx]
    for t in range(1, 4):
        z = z.maximum(a[idx + t * K])
    s = (z.reshape(M, K) @ inputs["b"].reshape(K, N)) * 0.25
    return {"o": (s @ inputs["c"].reshape(N, K) + z.reshape(M, K))
            .reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="matmul_reads_a_sibling_view", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "o_ptr": "*fp32", "I": "constexpr", "J": "constexpr",
               "K": "constexpr", "N": "constexpr", "M": "constexpr",
               "SI": "constexpr", "SJ": "constexpr"},
    constexprs={"I": I, "J": J, "K": K, "N": N, "M": M, "SI": SI, "SJ": SJ},
    args=[Arg("a", IN, "float32", (ASIZE,)),
          Arg("b", IN, "float32", (K * N,)),
          Arg("c", IN, "float32", (N * K,)),
          Arg("o", OUT, "float32", (M * K,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # The accumulation order is not torch's; same reason as tests/cases/smoke/gemm.py.
    rtol=1e-3, atol=1e-3,
    # THE COLLAPSE HAS TO STILL BE THERE for the kernel to pin anything: if
    # absorb_layout_ops takes it, no view reaches lower_to_vcix and this passes
    # vacuously.
    expect_ir={"08-transformed.mlir": {"memref.collapse_shape": 1,
                                   "torchsim.crosslane_move": 0}},
)
