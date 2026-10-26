"""A row gather and a row scatter whose indices the kernel COMPUTES, per program of a 2-D grid.

torch equivalent, program (x, y), p = x + 2 * y:
    o[p*M + (r*3 + p) % M] = t[(r*37 + 11*x + 5*y) % VOCAB]      for r in 0..M

THE INDICES EXIST ONLY IN THE FUNCTIONAL RUN. No input holds them: they are made
in the kernel from the program ids, so a timing model that reads input tensors,
or assumes 0, 1, 2, ..., cannot know them. TOGSim takes each indirect transfer's
offsets from the dump the vcix model wrote under the wrapper's dma_index_key, and
the TOG finds it by (the work item's key, its rank among that item's indirect
transfers) -- six programs, a different gather and scatter in each.

What breaks this: a TOG that carries the transfer as plain strided (togcheck:
`indirect=True` the TOG never emits), a dispatch key other than the wrapper's
(togcheck), or a dump TOGSim cannot find (--timing: "found no index dump").

Mechanism tier: indices computed per program, on both MVIN and MVOUT, across a 2-D grid.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

VOCAB = 1000
D = 32
M = 128          # rows per program == PYTORCHSIM_COMPILE_VECTORLANE_SIZE
GX, GY = 2, 3


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_index_computed_per_program_kernel(t_ptr, o_ptr, VOCAB: tl.constexpr,
                                                 D: tl.constexpr, M: tl.constexpr,
                                                 GX: tl.constexpr):
        x = tl.program_id(0)
        y = tl.program_id(1)
        p = x + GX * y
        rows = tl.arange(0, M)
        cols = tl.arange(0, D)
        src = (rows * 37 + 11 * x + 5 * y) % VOCAB
        dst = p * M + (rows * 3 + p) % M
        e = tl.load(t_ptr + src[:, None] * D + cols[None, :])
        tl.store(o_ptr + dst[:, None] * D + cols[None, :], e)

    return gather_index_computed_per_program_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"t": torch.randn(VOCAB * D, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    t = inputs["t"].reshape(VOCAB, D)
    o = torch.zeros(GX * GY * M, D, dtype=torch.float32)
    r = torch.arange(M)
    for y in range(GY):
        for x in range(GX):
            p = x + GX * y
            o[p * M + (r * 3 + p) % M] = t[(r * 37 + 11 * x + 5 * y) % VOCAB]
    return {"o": o.reshape(-1)}


SPEC = KernelSpec(
    name="gather_index_computed_per_program_kernel",
    kernel=kernel,
    signature={"t_ptr": "*fp32", "o_ptr": "*fp32", "VOCAB": "constexpr",
               "D": "constexpr", "M": "constexpr", "GX": "constexpr"},
    constexprs={"VOCAB": VOCAB, "D": D, "M": M, "GX": GX},
    args=[Arg("t", IN, "float32", (VOCAB * D,)),
          Arg("o", OUT, "float32", (GX * GY * M * D,))],
    grid=(GX, GY),
    reference=reference,
    make_inputs=make_inputs,
    rtol=0, atol=0,
)
