"""`other=` with a NON-ZERO value, and a store that does not hide it.

mask_structured_dims.py already loads with `other=-1.0`, but it masks the store
with the same predicate, so the filled lanes are never written out -- the fill
value is unobservable there and the kernel passes whatever the backend does with
it. This file is that kernel with the store mask REMOVED, so `o[n:]` is exactly
the fill and a wrong (or dropped) `other` shows up as a numeric mismatch.

WHAT IT PINS DOWN (backend, not the linalg lowering)

  * `other` is the LAST operand of tts.load, in its own segment
    (`operandSegmentSizes = [1, n_mask, n_other]`). Slicing the operand list
    from the tail by the dynamic-mask count therefore picks up the FILL where
    the EXTENT was meant to go. Before `_split_access_operands` read the
    segments, this kernel's transfer got

        masked_axes = [0], clamp operands = (%c0, %cst : f32)

    i.e. the f32 fill in the clamp slot and the real extent nowhere, and stage 4
    died on `'arith.index_cast' op operand #0 ... but got 'f32'`.

  * the fill reaches the DMA as a RAW BIT PATTERN, not as a number. Spike loads
    descriptor+136 as one uint64 and truncates it per element size
    (torchsim_mvin_common.h:206-235, `(uint32_t)desc_fill`), so -99.0f must
    arrive as 0xC2C60000 and -7:i32 as 0xFFFFFFF9. Both are checked here --
    the float branch and the integer branch of `_fill_bits`.

WHAT PROVES IT FIRED (04-transformed.mlir)
      masked_fill = 3267756032 : i64      (-99.0f  = 0xC2C60000)
      masked_fill = 4294967289 : i64      (-7:i32  = 0xFFFFFFF9)
  and the clamp operand next to each transfer is an `index`, not an f32.

torch equivalent: see `reference` -- x[:n] then the fill, per dtype.

Non-zero `other=` with an unmasked store, so the DMA fill is numerically observable. Covers both the float and the integer bit-pattern encoding of the descriptor fill field.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
NLIM = 96           # runtime `n`, deliberately not a multiple of BLOCK
FFILL = -99.0
IFILL = -7


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_other_fill_kernel(x_ptr, o_ptr, y_ptr, p_ptr, n,
                               BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        # the store is UNMASKED on purpose: the fill has to be observable.
        v = tl.load(x_ptr + r, mask=r < n, other=-99.0)
        tl.store(o_ptr + r, v)
        w = tl.load(y_ptr + r, mask=r < n, other=-7)
        tl.store(p_ptr + r, w)

    return mask_other_fill_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "y": torch.randint(-1000, 1000, (N,), generator=g,
                               dtype=torch.int32)}


def reference(inputs):
    import torch
    o = torch.full((BLOCK,), FFILL, dtype=torch.float32)
    o[:NLIM] = inputs["x"][:NLIM]
    p = torch.full((BLOCK,), IFILL, dtype=torch.int32)
    p[:NLIM] = inputs["y"][:NLIM]
    return {"o": o, "p": p}


SPEC = KernelSpec(
    name="mask_other_fill_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "y_ptr": "*i32",
               "p_ptr": "*i32", "n": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BLOCK,)),
          Arg("y", IN, "int32", (N,)), Arg("p", OUT, "int32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t")], "scalar_values": {"n": NLIM}},
)
