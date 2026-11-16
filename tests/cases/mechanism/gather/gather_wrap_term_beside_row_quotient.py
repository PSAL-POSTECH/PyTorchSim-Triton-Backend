"""SwinV2's head split, Inductor's kernel as it ran: `32*((x // 2048) % 3) + 6144*(x // 6144)` on a tile of two rows.

WHAT IT IS. `clone(transpose(view))` reads x0 + 32*x2 + 96*x1 + 6144*x3 with x2 = (x // 2048) % 3 and
x3 = x // 6144. A block of 4096 spans two rows of 2048, so PtrAnalysis makes x3 a gather with a 2-entry
index and keeps `32*x2` as a structured term that wraps at 96.

WHAT IT USED TO DO. On 48403b44: `DigitRefused: tensor.splat on a tile cut into digits` (the 2-entry index is
a replica, so its tile is cut), because materialize_layouts had no rule for the splat. With only that fixed
the values were wrong (max diff 6.6): folding the wrapped term onto the gather index dropped its `% 3`, so
row k read 32*k + 6144*(k // 3) instead of 32*(k % 3) + 6144*(k // 3).

WHAT WOULD BREAK THIS
  * no _cut_splat -- the refusal above
  * foldDimOntoOneIndex's flatten not applying the structured side's wrap -- wrong values from the second program

torch equivalent: o[x] = a[x % 32 + 32 * ((x // 2048) % 3) + 96 * ((x // 32) % 64) + 6144 * (x // 6144)]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N, XB = 786432, 4096


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_wrap_term_beside_row_quotient(in_ptr0, out_ptr0, XBLOCK: tl.constexpr):
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:]
        x0 = (xindex % 32)
        x1 = ((xindex // 32) % 64)
        x2 = ((xindex // 2048) % 3)
        x3 = xindex // 6144
        tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 96*x1 + 6144*x3), None)
        tl.store(out_ptr0 + (xindex), tmp0, None)
    return gather_wrap_term_beside_row_quotient


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(N, generator=g).to(torch.bfloat16), "o": torch.zeros(N, dtype=torch.bfloat16)}


def reference(inputs):
    import torch
    x = torch.arange(N)
    return {"o": inputs["a"][x % 32 + 32 * ((x // 2048) % 3) + 96 * ((x // 32) % 64) + 6144 * (x // 6144)].clone()}


SPEC = KernelSpec(
    name="gather_wrap_term_beside_row_quotient", kernel=kernel,
    signature={"in_ptr0": "*bf16", "out_ptr0": "*bf16", "XBLOCK": "constexpr"},
    constexprs={"XBLOCK": XB},
    args=[Arg("a", IN, "bfloat16", (N,)), Arg("o", OUT, "bfloat16", (N,))],
    grid=(N // XB,), reference=reference, make_inputs=make_inputs,
)
