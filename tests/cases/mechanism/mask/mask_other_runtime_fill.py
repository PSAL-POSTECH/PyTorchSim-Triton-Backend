"""`other=` whose value is only known at RUNTIME.

mask_other_fill.py covers the constant fill, which the backend can bake into the
DMA descriptor global's initial value. This file is the same kernel with the
fill turned into a kernel ARGUMENT, which the backend cannot bake: the
descriptor global is initialised at compile time, so the value has to be STORED
into it before CONFIG_DESC -- the same runtime-patch path the mask clamp already
uses for dim_low/dim_high.

WHAT IT PINS DOWN (backend)

  * the fill is an f32 but the descriptor field is an integer, so the value goes
    through `arith.bitcast f32 -> i32` + `extui -> i64` before it can be stored.
    Passing the operand through unconverted is the obvious wrong version.
  * the runtime value rides as ONE extra operand AFTER the clamp pairs, so
    stage 4's `mvals[2*i], mvals[2*i+1]` clamp slicing is unaffected; a
    `dyn_fill` unit attribute is what tells it the extra operand is there.
  * a runtime fill forces the UNIQUE i32-view descriptor global (the deduped
    static one cannot hold a runtime word) -- same as a masked or indirect DMA.

WHAT PROVES IT FIRED
  04-transformed.mlir  `arith.bitcast %arg3 : f32 to i32` + `arith.extui` feeding the
                   MVIN transfer's LAST operand, with `dyn_fill` set and
                   `masked_fill = 0` (the static field is unused).
  05-lowered.mlir  four stores into the i32-view descriptor before CONFIG_DESC --
                   words 7 and 11 are dim_low[3]/dim_high[3] (the rank-1 tile is
                   4D-expanded, so axis 0 lands at index 3) and words 34/35 are
                   the low and high halves of the fill at byte 136.

NOTE ON gap_fill.py
  That kernel also passes a runtime `other=` (it is the only way to reach
  `gsv-splat-walk`), but its mask is `r < BLOCK` -- a full-tile extent, which
  clamps to a no-op, so its fill is dead and the backend deliberately skips the
  descriptor store. It also does not compile past stage 3 for an unrelated
  reason (a scalar tts.gather). Hence this file.

torch equivalent: see `reference` -- x[:n] then the fill.

Runtime `other=`: the fill value is a kernel argument, so it is bitcast to an integer and stored into the DMA descriptor at runtime rather than baked into the global's initial value.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 1024
BLOCK = 128
NLIM = 96           # runtime `n`, deliberately not a multiple of BLOCK
FILL = -3.5         # runtime `fill`, exactly representable so err must be 0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_other_runtime_fill_kernel(x_ptr, o_ptr, n, fill,
                                       BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        # unmasked store: the fill has to be observable
        v = tl.load(x_ptr + r, mask=r < n, other=fill)
        tl.store(o_ptr + r, v)

    return mask_other_runtime_fill_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    o = torch.full((BLOCK,), FILL, dtype=torch.float32)
    o[:NLIM] = inputs["x"][:NLIM]
    return {"o": o}


SPEC = KernelSpec(
    name="mask_other_runtime_fill_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "fill": "fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("o", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("n", "int32_t"), ("fill", "float")],
           "scalar_values": {"n": NLIM, "fill": FILL}},
)
