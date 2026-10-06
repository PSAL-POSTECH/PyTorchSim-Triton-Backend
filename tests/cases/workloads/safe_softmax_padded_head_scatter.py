"""The same safe softmax, STORED where attention wants it: a padded head layout.

`safe_softmax_masked_tail.py` covers the two custom reductions and passes. This
one changes exactly one thing -- where the result goes -- and it is the half of
ViT's `triton_npu_fused__safe_softmax_view_10` that file does not reach.

The row index is flat over heads and queries, and the destination is not:

    x0 = xindex              a row of the [heads*S, S] view
    x2 = xindex % S          which query
    x3 = xindex // S         which head
    load   in_ptr0 + (r + S*x0)                  packed:  S*S per head
    store  out_ptr + (r + S*x2 + PITCH*x3)       padded:  PITCH per head

`PITCH` is what `empty_strided` gave the tensor -- 38816 against a natural
38809 for S = 197 -- so the destination has a SEVEN-ELEMENT HOLE at the end of
every head and the two index expressions cannot be the same affine map. One
program's XBLOCK of rows straddles a head boundary (2364 rows in blocks of 128
does not divide by 197), so `%` and `//` of the program's own index are what
place every element.

WHAT IT IS FOR. A `%`/`//` pair on the program index, feeding a scatter into a
padded destination, is the class this backend has been wrong on before -- see
mixed_wrap_transposed_a.py, where an identity `%` on a matmul's row index
changed every element, and the DeepSeek gather "an arange and two moduli get
wrong". Here it is not a wrap at all: nothing is out of range, and the modulus
is doing arithmetic rather than guarding.

MEASURED ON THE MODEL, which is the reason this file exists:

    [FuncVerify] first divergence at buffer 'buf18' (op aten.expand.default)
                 max abs diff 0.0641784, 416920/465708 elements over tol

-- 89% of the elements of a softmax, so it is not a corner. buf14, the input,
verifies, and the reductions verify on their own next door; what is left is the
address.

HEADS = 3 rather than 12 so a divergence is readable by hand, and it is still
more than one so the head stride is exercised. The row count 3*197 = 591 is not
a multiple of the lane count, which is what keeps the xmask real.

Inductor's attention softmax whole: two custom reductions and a %/// scatter of the program index into a padded head layout. The reductions alone pass in safe_softmax_masked_tail.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

HEADS, S = 3, 197
BLOCK = 256
PITCH = 38816                 # what empty_strided gives [.., 197, 197]
ROWS = HEADS * S


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _maximum(a, b):
        mask = (a > b) | (a != a)
        return tl.where(mask, a, b)

    @triton.jit
    def _any_combine(a, b):
        return a | b

    @triton.jit
    def safe_softmax_padded_head_scatter(
            x_ptr, o_ptr, ROWS: tl.constexpr, S: tl.constexpr,
            PITCH: tl.constexpr, XBLOCK: tl.constexpr, BLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        x0 = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = x0 < ROWS
        rn = tl.arange(0, BLOCK)[None, :]
        rmask = rn < S
        x2 = x0 % S
        x3 = x0 // S
        x = tl.load(x_ptr + (rn + S * x0), rmask & xmask, other=0.0)

        finite = tl.reduce(tl.where(rmask & xmask, x != float("-inf"), False),
                           1, _any_combine)[:, None]
        mx = tl.reduce(tl.where(rmask & xmask, x, float("-inf")), 1,
                       _maximum)[:, None]
        e = tl.exp(x - mx)
        s = tl.sum(tl.where(rmask & xmask, e, 0.0), 1)[:, None]
        # THE SCATTER. Not where it was read from: a padded head pitch.
        tl.store(o_ptr + (rn + S * x2 + PITCH * x3),
                 tl.where(finite, e / s, 0.0), rmask & xmask)

    return safe_softmax_padded_head_scatter


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, S, generator=g, dtype=torch.float32)
    ninf = float("-inf")
    x[0, :] = ninf                      # a row with no finite element
    x[S + 1, :] = ninf                  # and one in the second head
    x[S + 1, S - 1] = 1.0
    return {"x": x}


def reference(inputs):
    import torch
    x = inputs["x"]
    finite = torch.isfinite(x).any(dim=1, keepdim=True)
    e = torch.exp(x - x.amax(dim=1, keepdim=True))
    soft = torch.where(finite, e / e.sum(dim=1, keepdim=True),
                       torch.zeros_like(e))
    # The padded destination: head h occupies [h*PITCH, h*PITCH + S*S), and the
    # tail of each head is never written.
    out = torch.zeros(HEADS * PITCH, dtype=torch.float32)
    for h in range(HEADS):
        out[h * PITCH:h * PITCH + S * S] = soft[h * S:(h + 1) * S].reshape(-1)
    return {"o": out}


SPEC = KernelSpec(
    name="safe_softmax_padded_head_scatter", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "ROWS": "constexpr",
               "S": "constexpr", "PITCH": "constexpr", "XBLOCK": "constexpr",
               "BLOCK": "constexpr"},
    constexprs={"ROWS": ROWS, "S": S, "PITCH": PITCH, "XBLOCK": 128,
                "BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (ROWS, S)),
          Arg("o", OUT, "float32", (HEADS * PITCH,))],
    grid=((ROWS + 127) // 128,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
