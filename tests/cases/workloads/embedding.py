"""Embedding lookup with a scale -- a tile whose ROW BASE is loaded from memory.

torch equivalent: (table[idx] * scale[:, None])

WHY THIS SHAPE IS DIFFERENT FROM EVERY OTHER GATHER HERE.  tests/cases/
gathers scatter WITHIN a row: the offsets are a rank-1 tensor and the tile stays
one strip. Here the index tensor selects whole ROWS of a table:

    row = tl.load(idx_ptr + rows)                     # i32, one per lane
    e   = tl.load(t_ptr + row[:, None] * D + cols[None, :])

so the address is affine in the in-lane axis (`cols`, stride 1) and completely
unstructured in the LANE axis. That is the one combination the DMA has an
instruction for and the compiler has to recognise: `indirect_mode` reads one
index per lane and adds `idx * indirect_stride` to that lane's base, which is
exactly `row[:, None] * D`. A tile that were unstructured the other way round --
scattered in-lane, affine across lanes -- has no such form.

THE INDEX TENSOR IS THE INPUT THAT MATTERS. It repeats (several lanes read the
same table row) and it is not sorted, so nothing about the access is contiguous
and a compiler that quietly treated it as a stride would still produce plausible
numbers on a sorted index. It is generated with duplicates on purpose.

`scale` is a second rank-1 input broadcast along the in-lane axis, so the gather
result is immediately consumed by a per-row broadcast -- the softmax shape --
which forces the gathered tile and a normal rank-1 tile into the same component
of the lane-axis forest.

Complex-op tier: a row-indirect gather -- unstructured across the lanes, affine inside one -- feeding a per-row broadcast.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

VOCAB = 512      # table rows
D = 64           # embedding dim, in-lane
M = 128          # tokens == PYTORCHSIM_COMPILE_VECTORLANE_SIZE, one program


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def embedding_kernel(idx_ptr, t_ptr, scale_ptr, o_ptr,
                         D: tl.constexpr, M: tl.constexpr):
        rows = tl.arange(0, M)
        cols = tl.arange(0, D)
        row = tl.load(idx_ptr + rows)
        e = tl.load(t_ptr + row[:, None] * D + cols[None, :])
        s = tl.load(scale_ptr + rows)
        tl.store(o_ptr + rows[:, None] * D + cols[None, :], e * s[:, None])

    return embedding_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Deliberately repeating and unsorted: a stride would still fit a sorted one.
    idx = torch.randint(0, VOCAB, (M,), generator=g, dtype=torch.int32)
    return {
        "idx": idx,
        "t": torch.randn(VOCAB * D, generator=g, dtype=torch.float32),
        "scale": torch.randn(M, generator=g, dtype=torch.float32),
    }


def reference(inputs):
    t = inputs["t"].reshape(VOCAB, D)
    e = t[inputs["idx"].long()]
    return {"o": (e * inputs["scale"][:, None]).reshape(-1)}


SPEC = KernelSpec(
    name="embedding_kernel",
    kernel=kernel,
    signature={"idx_ptr": "*i32", "t_ptr": "*fp32", "scale_ptr": "*fp32",
               "o_ptr": "*fp32", "D": "constexpr", "M": "constexpr"},
    constexprs={"D": D, "M": M},
    args=[
        Arg("idx", IN, "int32", (M,)),
        Arg("t", IN, "float32", (VOCAB * D,)),
        Arg("scale", IN, "float32", (M,)),
        Arg("o", OUT, "float32", (M * D,)),
    ],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
