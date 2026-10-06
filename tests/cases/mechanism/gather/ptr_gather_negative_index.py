"""A gather whose index is NEGATIVE, in the four-byte field where that was wrong.

The descriptor's `indirect_esize` is a byte count and the DMA dispatches on it,
loading with `load_uint8/16/32/64`. THE FIRST THREE ARE ZERO EXTENSIONS of a
value the program means as signed, so an index of -128 in a four-byte field
reads as 4294967168 and `base + idx * scale` lands 2^34 past the tile. At eight
bytes the zero extension is the identity and two's complement carries the
multiply and the add, so i64 was always right -- which is exactly why the
compiler used to widen EVERY index tile to i64, at a buffer and a copy per
transfer.

spike `df08f17` reads it signed, and dropping the widening is what that bought.
The branch this compiler pins is named after it -- but versions.env recorded two
OTHER reasons for the pin and not this one, and nothing in the suite held it.

    base = x_ptr + N/2        so a negative index still lands in the buffer
    idx  = i - N/2            half the lanes are negative, half are not

THE INDEX IS DATA, so this is a gather and the values are never folded: PtrAnalysis
carries the loaded tensor into gather_scatter_offset and no arithmetic here reads
its sign. The sign lives at the descriptor, which is why a spike is the only
thing that can get it wrong and why the kernel has to run to say so.

BOTH SIGNS IN ONE TILE ON PURPOSE. An all-negative index would also read wrong
under an unsigned load, but a mixed one fails on exactly the lanes that are
negative -- so the error names the half rather than the whole.

torch equivalent: o = x[N/2 + (i - N/2)] == x[i]

Half the gather's indices are negative, in an i32 tile -- the width the zero extension broke. Under a stock spike those 32 lanes read 4294967168-ish offsets; the pin is df08f17 and versions.env now says so.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_negative_index_kernel(x_ptr, i_ptr, o_ptr, N: tl.constexpr):
        r = tl.arange(0, N)
        idx = tl.load(i_ptr + r)               # runs from -N/2 to N/2 - 1
        v = tl.load(x_ptr + N // 2 + idx)
        tl.store(o_ptr + r, v)

    return ptr_gather_negative_index_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "i": (torch.arange(N, dtype=torch.int32) - N // 2)}


def reference(inputs):
    import torch
    idx = inputs["i"].long() + N // 2
    return {"o": inputs["x"][idx]}


SPEC = KernelSpec(
    name="ptr_gather_negative_index_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("i", IN, "int32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # ONE INDIRECT TRANSFER, so the sign question reaches the descriptor at all.
    expect_ir={"04-transformed.mlir": {"offset_stride": 1}},
)
