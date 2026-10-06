"""A grid offset that must go INSIDE the `%`, on a wrapped axis whose stride is not 1.

WHAT BREAKS THIS. Upstream PtrAnalysis records `(x) % m` by writing the modulus
into `shape[d]` of the non-singleton dim d. The offset the program added BEFORE
the `%` has to be under that modulus too -- but Inductor expands the tensor to
rank 2 first, so by the time the remainder is visited the grid offset has already
been parked on the OTHER dim by visitOperandSplat ("the offset of the outer most
dimension" is an address claim, not a statement about dim 0). It then rides
outside the wrap and is added to the address raw.

WHY THE SHAPE HAS TO BE THIS ONE:

  - `x % WRAP` where x carries a per-program offset. With a 1-program grid the
    offset is 0 and hasConstZero hides the bug -- hence grid (NX // BX,) = 16.
  - NX = 2 * WRAP, so programs 8..15 are exactly the ones whose offset must fold
    back to the start. Anything smaller never crosses the seam.
  - the wrapped axis is scaled by ROW, so the modulus arrives pre-scaled as
    WRAP * ROW = 2048 and the stray offset arrives scaled the same way. This is
    rule 18's "wrap range is in units of that axis" from the other side: the
    number is right, it is just under no modulo at all.
  - the y axis has stride 1 and its own offset, so the two axes cannot be told
    apart by "which one has an offset".

WHAT IT LOOKED LIKE. This is Llama's RoPE, `cat([freqs, freqs], -1)`, which
Inductor folds into one load with `x1 % 64`. The second half read past the row
instead of back around it, came back as angle 0, and cos/sin returned 1/0 --
max abs diff 0.5288 against CPU with row 0 exact and every other row wrong.

INPUT IS TWICE THE WRAP so the failure is a wrong number and not a fault: the
stray offset lands inside the buffer and reads real values from the wrong place.
A buffer sized exactly to the wrap would make this an out-of-bounds read, which
is a different and less useful signal.

grid offset must stay under the modulo on a strided wrapped axis
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BY, BX = 32, 8
WRAP, ROW = 64, 32               # modulus and the stride it is scaled by
NX = 2 * WRAP                    # 128 -> half the grid must fold back
NIN = 2 * WRAP * ROW             # 4096, twice the wrap: wrong reads stay in bounds


def kernel():
    import triton, triton.language as tl

    @triton.jit
    def ptr_wrap_scalar_offset_on_singleton_dim_kernel(
            x_ptr, o_ptr, BY: tl.constexpr, BX: tl.constexpr,
            WRAP: tl.constexpr, ROW: tl.constexpr, NX: tl.constexpr):
        xoff = tl.program_id(0) * BX
        y = tl.arange(0, BY)[:, None]
        x = xoff + tl.arange(0, BX)[None, :]
        v = tl.load(x_ptr + (y + ROW * (x % WRAP)))
        tl.store(o_ptr + (y * NX + x), v)

    return ptr_wrap_scalar_offset_on_singleton_dim_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NIN, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    y = torch.arange(BY)[:, None]
    x = torch.arange(NX)[None, :]
    return {"o": inputs["x"][y + ROW * (x % WRAP)].reshape(-1)}


SPEC = KernelSpec(
    name="ptr_wrap_scalar_offset_on_singleton_dim_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "BY": "constexpr",
               "BX": "constexpr", "WRAP": "constexpr", "ROW": "constexpr",
               "NX": "constexpr"},
    constexprs={"BY": BY, "BX": BX, "WRAP": WRAP, "ROW": ROW, "NX": NX},
    args=[Arg("x", IN, "float32", (NIN,)), Arg("o", OUT, "float32", (BY * NX,))],
    grid=(NX // BX,), reference=reference, make_inputs=make_inputs,
    )
