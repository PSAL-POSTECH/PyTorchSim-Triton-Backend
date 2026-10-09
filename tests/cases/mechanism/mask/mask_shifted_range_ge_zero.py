"""A range shifted NEGATIVE and masked with `>= 0`: two spellings, two answers.

    a = tl.load(x_ptr + (t - SHIFT),    mask=(t - SHIFT) >= 0,    other=FILL)
    b = tl.load(x_ptr + ((-SHIFT) + t), mask=((-SHIFT) + t) >= 0, other=FILL)

The two lines say the same thing and must agree. MEASURED on the linalg lowering
@ef6ebef, they do not -- `a` is right and `b` is wrong:

    t - SHIFT       arith.subi   MaskAnalysis has no parseSub, the mask parse
                                 FAILS, and the access falls to tts.gather with
                                 a real i1 mask and the fill. Right answer.
    (-SHIFT) + t    arith.addi   parseAdd succeeds and makes MaskState::start
                                 -5, then parseCmp folds `>= 0` away as
                                 always-true: static_mask_dims = [64] on a
                                 base_offset of -5. The descriptor reads
                                 x[-5..-1] and the fill never happens.

THE FOLD IS RIGHT ABOUT LANES AND WRONG ABOUT VALUES. MaskAnalysis.cpp:548 says
"Dims already has an implicit assumption that the lower bound is 0", and that
assumption is about the LANE index. `value >= 0` is `lane >= 0` only while the
range starts at or above zero; here the range starts at -5, so it is
`lane >= 5`. dims is a single number counting from the front of the tile and
cannot say that, so the comparison is dropped instead of refused.

THE SUPPORTED SPELLING IS THE BROKEN ONE, which is why this had gone unseen: a
NON-ZERO bound (`>= 1`) is refused and comes out right down the elementwise path
(mask_1d_parse_reject.py pins that refusal). Only `>= 0`, the one comparison
MaskAnalysis claims to fold, reaches the bug.

WHERE IT COMES FROM. `(-K) + x` is what Inductor emits whenever an output draws
from a shifted source region -- mask_negative_offset.py quotes the shape from
torch.cat: `tl.load(in_ptr1 + (x0 + 16*((-8) + x1)), tmp6 & xmask, other=0.0)`.
It is also the left pad of every causal convolution.

torch equivalent: o[k] = x[k-SHIFT] for k >= SHIFT, FILL otherwise, twice.

`t - K` and `(-K) + t` masked with `>= 0` must agree. They do not: parseAdd makes MaskState::start negative and parseCmp folds the comparison away, so the descriptor reads in front of the buffer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BLOCK = 64
SHIFT = 5
FILL = -7.0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_shifted_range_ge_zero(x_ptr, a_ptr, b_ptr, BLOCK: tl.constexpr,
                                   SHIFT: tl.constexpr, FILL: tl.constexpr):
        t = tl.arange(0, BLOCK)
        # arith.subi -- no parseSub, so the mask survives as an i1 tensor
        sub = t - SHIFT
        tl.store(a_ptr + t,
                 tl.load(x_ptr + sub, mask=sub >= 0, other=FILL))
        # arith.addi with a negative constant -- parseAdd succeeds and the
        # `>= 0` is folded away against a range that starts below zero
        add = (-SHIFT) + t
        tl.store(b_ptr + t,
                 tl.load(x_ptr + add, mask=add >= 0, other=FILL))

    return mask_shifted_range_ge_zero


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(BLOCK, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    o = torch.full((BLOCK,), FILL, dtype=torch.float32)
    o[SHIFT:] = inputs["x"][:BLOCK - SHIFT]
    return {"a": o, "b": o.clone()}


SPEC = KernelSpec(
    name="mask_shifted_range_ge_zero", kernel=kernel,
    signature={"x_ptr": "*fp32", "a_ptr": "*fp32", "b_ptr": "*fp32",
               "BLOCK": "constexpr", "SHIFT": "constexpr", "FILL": "constexpr"},
    constexprs={"BLOCK": BLOCK, "SHIFT": SHIFT, "FILL": FILL},
    args=[Arg("x", IN, "float32", (BLOCK,)), Arg("a", OUT, "float32", (BLOCK,)),
          Arg("b", OUT, "float32", (BLOCK,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    expect_ir={"07-regularized.mlir": {"tts.gather": 0}},
)
