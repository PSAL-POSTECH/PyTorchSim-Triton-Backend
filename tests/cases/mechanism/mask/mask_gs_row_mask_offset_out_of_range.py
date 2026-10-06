"""A GATHER POINTER's row mask, and the row indices it used to leave live.

    was: User load segfault @ 0x00000000d9001010

One index per ROW, so PtrAnalysis builds a tts.make_gather_scatter_tptr rather
than a tts.gather, and the mask rides the GATHERED axis of that pointer. The
masked-off rows hold indices far out of range -- that is the whole shape of the
kernel.

WHAT USED TO HAPPEN. _rewrite_gs_mask rebuilt the pointer WITHOUT its mask and
put the mask in a select over the loaded rows. Every masked-off index stayed
live in the descriptor, and the indirect DMA dereferenced each one.

WHAT HAPPENS NOW. The row indices go to 0 before the mask comes off the
pointer, which is the answer _rewrite_gather already gave for a per-element
gather ("an offset under a mask is not read, so it must not be dereferenced
either; zero is safe because nothing here WRITES"). The select still runs, for
the value.

WHY 0 AND NOT THE PADDING. emit_wrapper_c doubles `out`/`inout` so a masked-off
STORE has somewhere to land; an `in` gets alignment only, because Llama 4
Scout's 1.25 GiB expert stack cannot be asked for twice. A read does not need
the room -- it may land anywhere in bounds, and the select discards it.

THE SIBLING is mask_unstructured_read_leaves_the_buffer, the same root cause on
a STRIDED pointer where the address is a stride rather than an index. Both
segfaulted and the fix was one decision, not two.

torch equivalent: rows where flag is false read nothing and come back 0.

A gather tile pointer whose masked-off ROW indices are far out of range. Pins that the mask reaches the index, not only the values, on the gs-tptr route as well as the tts.gather one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS = 8
COLS = 16
# Far past x's 512-byte round-up, so a masked-off row is not quietly absorbed.
OFF = 1 << 18


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_gs_row_mask_offset_out_of_range_kernel(
            x_ptr, f_ptr, o_ptr, OFF: tl.constexpr,
            ROWS: tl.constexpr, COLS: tl.constexpr):
        r = tl.arange(0, ROWS)
        c = tl.arange(0, COLS)
        # Loaded, so MaskAnalysis cannot fold this into an extent.
        flag = tl.load(f_ptr + r) != 0
        # The index is its own expression: `flag` says nothing about its size.
        idx = tl.where(flag, r, OFF)
        p = x_ptr + idx[:, None] * COLS + c[None, :]
        v = tl.load(p, mask=flag[:, None], other=0.0)
        tl.store(o_ptr + r[:, None] * COLS + c[None, :], v)

    return mask_gs_row_mask_offset_out_of_range_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.zeros(ROWS, dtype=torch.int32)
    f[::2] = 1                                   # half the rows are masked off
    return {"x": torch.randn(ROWS * COLS, generator=g, dtype=torch.float32),
            "f": f}


def reference(inputs):
    import torch
    x, f = inputs["x"].reshape(ROWS, COLS), inputs["f"] != 0
    return {"o": torch.where(f[:, None], x, torch.zeros_like(x))}


SPEC = KernelSpec(
    name="mask_gs_row_mask_offset_out_of_range_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "f_ptr": "*i32", "o_ptr": "*fp32",
               "OFF": "constexpr", "ROWS": "constexpr", "COLS": "constexpr"},
    constexprs={"OFF": OFF, "ROWS": ROWS, "COLS": COLS},
    args=[Arg("x", IN, "float32", (ROWS * COLS,)),
          Arg("f", IN, "int32", (ROWS,)),
          Arg("o", OUT, "float32", (ROWS, COLS))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
