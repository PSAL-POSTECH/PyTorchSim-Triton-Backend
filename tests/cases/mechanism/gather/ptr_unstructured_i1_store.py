"""Storing a PREDICATE makes the whole function unstructured -- and every
pointer op in it arrives as a linalg.generic over a tensor of `!tt.ptr`.

WHY A BOOL STORE DOES THIS. PtrAnalysis measures a pointer in ELEMENTS and an
i1 has no byte stride to measure in, so `tl.store(p + idx, x <= 0.0)` is one it
cannot structure. TritonToUnstructured is per FUNCTION, so the one access it
gives up on takes every other access with it: the f32 load beside it, the
broadcast row, the f32 store. Twelve lines reproduce it --

    tl.store(out_ptr2 + idx, x <= 0.0)
    -> _Fatal: the base of a tt.addptr generic is not an i64 tile:
       ins(%20, %14 : tensor<8x8x!tt.ptr<i1>>, tensor<8x8xi32>)

-- and nothing else in Inductor's `mm` template does. Its `rm % M`, its
`max_contiguous`/`multiple_of` hints, its K loop, its transposed B index and its
`tl.dot` all structure cleanly; the reduction was run with each of those in turn
and only the predicate store failed.

WHAT THIS PINS, four things, each of which was uncovered by fixing the one
before it:

    the SPLAT the chain starts as    `linalg.fill ins(%base : !tt.ptr<f32>)`
                                     computes no address -- one value in every
                                     element -- and was skipped, so the addptr
                                     below it was refused for a base nothing
                                     would ever have dissolved
    the BROADCAST of a row           a generic whose body is only its yield
                                     moves addresses where its MAPS say and
                                     changes none of them (the `c` load here)
    the MASK, both ways              a tensor tt.load/tt.store carrying one i1
                                     per element was refused outright; it goes
                                     to tts.gather / tts.scatter, whose mask
                                     operands _empty_read_mask_carriers already reads back
    the ANCHOR's width is BYTES      `memref<*xi1>` under an i8 access is one
                                     byte either way, and comparing elem_BITS
                                     found no argument to anchor the gather on

AND THE STEP THAT MADE IT COMPILE AND STILL BE WRONG. `tt.addptr` advanced by
`elem_bits(pointee) // 8`, which is ZERO for i1, so every address in the
predicate tile collapsed onto the base: 8x8 elements all scattered to index 0
and the buffer came back zero. It compiled, lowered, linked and ran to say it --
`o` was exact the whole time and only `b` was wrong, 23 of 48 (rule 12). The
step is `elem_bytes` now, one byte, the same number the DMA descriptor and
bufferize already used.

WHY THE VALUES ARE EXACT. Every address here is computed by this backend rather
than described to the DMA, so a wrong one shows up as a wrong number and not as
a refusal: rtol = atol = 0, and the predicate output is the half that catches an
i1 written a byte apart from where it is read.

NO `other=` HERE, and that is a refusal rather than an omission. Triton's fill
is a TILE on an unstructured load and tts.gather's `other` slot is the POINTEE
type -- a scalar, by its own ODS -- so there is nowhere to put it and
convert_unsupported_pointers says so by name. Masked-off elements are left undefined,
which is what Triton promises without `other=`, and _empty_read_mask_carriers fills them
with zero. Inductor's kernel 9 is written the same way (its bias load carries
`eviction_policy` and no fill), so nothing in front of this needs the tile form
yet; when something does, the fix is a select against the tile and it belongs
in _empty_read_mask_carriers, not here.

WHAT WOULD BREAK THIS. Dissolving the fill but not the broadcast (or the other
way round); dropping either mask instead of carrying it; comparing element bits
instead of bytes at the anchor; or reading the broadcast by op NAME rather than
from its maps and its empty body.

A predicate store, which PtrAnalysis cannot structure, so every access in the function arrives as a tensor of pointers. Covers convert_unsupported_pointers' splat and address-move arms, the masks it hands to tts.gather / tts.scatter, and the byte-width anchor for an i1 argument.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M = 6           # real rows -- the mask is M < BM
N = 8           # real columns
BM = 8
BN = 8


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def unstructured_i1_store(x_ptr, c_ptr, o_ptr, b_ptr,
                              M: tl.constexpr, N: tl.constexpr,
                              BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)[:, None]
        rn = tl.arange(0, BN)[None, :]
        mask = (rm < M) & (rn < N)
        idx = rn + N * rm
        x = tl.load(x_ptr + idx, mask)
        # A ROW of addresses, carried across the tile: the second thing that
        # holds a pointer tile without computing one.
        c = tl.load(c_ptr + tl.broadcast_to(rn, [BM, BN]), mask)
        y = x + c
        tl.store(o_ptr + idx, y, mask)
        # THE PREDICATE STORE, which is why none of the above is structured.
        tl.store(b_ptr + idx, y <= 0.0, mask)

    return unstructured_i1_store


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(M * N, generator=g, dtype=torch.float32),
            "c": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    y = inputs["x"].reshape(M, N) + inputs["c"][None, :]
    return {"o": y.reshape(-1).contiguous(),
            "b": (y <= 0.0).reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="unstructured_i1_store", kernel=kernel,
    signature={"x_ptr": "*fp32", "c_ptr": "*fp32", "o_ptr": "*fp32",
               "b_ptr": "*i1", "M": "constexpr", "N": "constexpr",
               "BM": "constexpr", "BN": "constexpr"},
    constexprs={"M": M, "N": N, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (M * N,)),
          Arg("c", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (M * N,)),
          Arg("b", OUT, "bool", (M * N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=0, atol=0,
)
