"""4-bit weights with group scales -- AWQ/GPTQ unpack, then the matmul.

torch equivalent:
    w  = ((packed >> (4 * (j % 2))) & 0xF) - 8      # two nibbles per int8
    w  = w * scale[group_of(j)]                      # per-GROUP dequant
    out = x @ w

WHY EVERY OPEN-WEIGHT MODEL SHIPS LIKE THIS
  A 7B model is 14GB in fp16 and 4GB at 4 bits, which is the difference between
  fitting and not. The weights are stored packed two-per-byte with one scale per
  group of G output elements, and the dequant happens INSIDE the matmul kernel
  because materialising the fp16 weight defeats the point.

THREE THINGS THAT ARE EACH A DIFFERENT KIND OF HARD
  1. THE UNPACK IS BIT ARITHMETIC ON THE TILE. A shift whose amount alternates
     with position, then a mask, then a bias by 8 to recover the sign. Nothing
     here is elementwise-uniform: the shift depends on the index.
  2. THE SCALE IS A BROADCAST FROM A COARSER AXIS. One scale serves G weights,
     so the operand has to be stretched along the reduction axis -- the exact
     shape sort_radix.py already showed this backend refuses at rank 3, asked
     here at rank 2 where it should be reachable.
  3. THE PRODUCT IS MIXED PRECISION. Weights arrive as int8, become int, scale
     to float, and only then multiply the activations. The dtype changes twice
     inside one expression.

GROUP SIZE 8, not the usual 128, so the whole thing fits a tile that can be read
in the IR. The arithmetic is identical; only the ratio changes.

Index-dependent shift, a coarse-axis broadcast, and two dtype changes, all before the dot. The dequant is the kernel; the matmul is the easy half.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, K, N, G = 16, 32, 16, 8            # groups run along K


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def quant_int4_matmul_kernel(x_ptr, wq_ptr, s_ptr, o_ptr,
                                 M: tl.constexpr, K: tl.constexpr,
                                 N: tl.constexpr, G: tl.constexpr):
        rm = tl.arange(0, M)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)

        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])            # [M, K]

        # Two nibbles per byte along K: element k lives in byte k//2, in the
        # low nibble when k is even. The SHIFT DEPENDS ON THE INDEX.
        byte = tl.load(wq_ptr + (rk[:, None] // 2) * N + rn[None, :])
        shift = (rk[:, None] % 2) * 4
        w = ((byte >> shift) & 0xF) - 8                               # [K, N]

        # One scale per group of G along K, stretched back over the group.
        s = tl.load(s_ptr + (rk[:, None] // G) * N + rn[None, :])
        wf = w.to(tl.float32) * s

        tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(x, wf))

    return quant_int4_matmul_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M, K, generator=g, dtype=torch.float32),
            # Packed nibbles: byte b holds elements 2b (low) and 2b+1 (high).
            "wq": torch.randint(0, 256, (K // 2, N), generator=g,
                                dtype=torch.int32),
            "s": torch.rand(K // G, N, generator=g, dtype=torch.float32) * 0.1}


def reference(inputs):
    import torch
    byte = inputs["wq"]
    lo = (byte & 0xF) - 8
    hi = ((byte >> 4) & 0xF) - 8
    w = torch.stack([lo, hi], dim=1).reshape(K, N).to(torch.float32)
    w = w * inputs["s"].repeat_interleave(G, dim=0)
    return {"o": inputs["x"] @ w}


SPEC = KernelSpec(
    name="quant_int4_matmul_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "wq_ptr": "*i32", "s_ptr": "*fp32",
               "o_ptr": "*fp32", "M": "constexpr", "K": "constexpr",
               "N": "constexpr", "G": "constexpr"},
    constexprs={"M": M, "K": K, "N": N, "G": G},
    args=[Arg("x", IN, "float32", (M, K)),
          Arg("wq", IN, "int32", (K // 2, N)),
          Arg("s", IN, "float32", (K // G, N)),
          Arg("o", OUT, "float32", (M, N))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
