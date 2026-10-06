"""A SCALAR load from a bool buffer: the bitcast breaks the pointer chain.

    tmp1 = tl.load(in_ptr0 + (0)).to(tl.int1)

One element, from a `torch.bool` argument, broadcast over the tile. Triton reads
i1 through i8 because that is how a bool sits in memory, so the pointer goes

    !tt.ptr<i1>  --tt.bitcast-->  !tt.ptr<i8>  --tt.load-->  i8

and `TritonToUnstructured` walks the pointer chain op by op, rewriting each into
offset arithmetic. Its case list had no entry for `tt.bitcast`, so the walk
STOPPED there -- printing `unexpected op in ptr sequence` and, because that
error is not fatal, carrying on. The load behind the bitcast was then never
converted, and a scalar `tt.load` on a `!tt.ptr<i8>` reached bufferize:

    no lowering for 'tt.bitcast': ... %12 = "tt.bitcast"(%0)
                                        : (!tt.ptr<i1>) -> !tt.ptr<i8>

which names the bitcast rather than the load stranded behind it. With the walk
taught that a pointer bitcast keeps the offset and moves only the type (and the
BASE with it, since `tts.gather`'s ptr operand is the scalar argument), the same
access becomes

    %58 = "tts.gather"(%57, %6) : (!tt.ptr<i8>, i32) -> i8

which convert_accesses_to_transfer already handles.

WHERE IT CAME FROM. Mistral's decode, kernel 21
(`_safe_softmax_add_scalar_tensor_view_where_21`) -- the same causal-mask
softmax as kernel 6, one decoded token later, where the query length is 1 and
the mask has collapsed to a single element. Kernel 6 compiles without the fix;
this one does not, which is why the reduction had to come from here. The model
went 21 kernels to 43 and its three decoded tokens all verify.

WHY A SCALAR AND NOT A TILE. `ptr_bitcast_bool_tile.py` covers the tile
spelling, and that one takes a different road: its offsets are not analysable at
all, so the load SCALARISES into a linalg.generic and convert_unsupported_pointers rewrites it. Nothing
in the suite had the shape where the chain is analysable and the bitcast is the
only thing in the way -- three attempts to reduce it from the tile side all
passed with and without the fix, pinning nothing, because they kept landing on
the other road.

WHAT WOULD BREAK IT: dropping the bitcast case from the walk, or moving only the
tensor type through it and leaving the base -- `tts.gather` then holds an i1 base
with the i8 the load actually reads, and its own verifier says so ("other
matches ptr type").

One element of a torch.bool buffer, read through the i1->i8 bitcast. Reduced from Mistral's kernel 21, where the pointer walk stopped at the bitcast and left the load behind it unconverted.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

ROWS, N = 8, 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def bitcast_bool_scalar_load(x_ptr, m_ptr, o_ptr, ROWS: tl.constexpr,
                                 N: tl.constexpr):
        row = tl.arange(0, ROWS)[:, None]
        col = tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + (col + N * row))
        # THE WHOLE OF IT: one element of a bool buffer, read through the
        # i1 -> i8 bitcast triton puts in front of every bool pointer.
        keep = tl.load(m_ptr + 0).to(tl.int1)
        # A FINITE fill: -inf minus -inf is nan and the checker would then
        # report a nan error over a kernel that verified nothing.
        tl.store(o_ptr + (col + N * row), tl.where(keep, x, -1e30))

    return bitcast_bool_scalar_load


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(ROWS * N, generator=g, dtype=torch.float32),
            "m": torch.ones(1, dtype=torch.bool)}


def reference(inputs):
    import torch
    x = inputs["x"]
    keep = bool(inputs["m"][0])
    return {"o": x if keep else torch.full_like(x, -1e30)}


SPEC = KernelSpec(
    name="bitcast_bool_scalar_load", kernel=kernel,
    signature={"x_ptr": "*fp32", "m_ptr": "*i1", "o_ptr": "*fp32",
               "ROWS": "constexpr", "N": "constexpr"},
    constexprs={"ROWS": ROWS, "N": N},
    args=[Arg("x", IN, "float32", (ROWS * N,)),
          Arg("m", IN, "bool", (1,)),
          Arg("o", OUT, "float32", (ROWS * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
)
