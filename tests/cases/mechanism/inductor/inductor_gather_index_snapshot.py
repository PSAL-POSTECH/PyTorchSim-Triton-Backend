"""Two gathers whose index tiles share a buffer, the second computed FROM the
first in place -- and the snapshot copy between them that must survive.

WHAT IT PINS. A gather's index tile is built into a buffer the next access
reuses, so bufferization snapshots it per access: a `torchsim.transfer` is
unregistered and the tensor world cannot see that it reads the snapshot.
`_write_where_the_copy_lands` removes such a copy when the write that feeds it
can simply land in the destination instead -- and its guard on what may touch
the source afterwards asked only whether the next use WRITES it:

    if after and not _writes_through(after[0][1], after[0][2]):
        continue

The next use here is the generic that builds the SECOND index, `idx2 = idx1 +
S*x1`, which has the buffer as both its `ins` and its `outs`. It writes it, so
the guard passed -- and it READS it first, which is the value that had just been
moved somewhere else. The buffer is then never written before that read:

    %alloc_2 = memref.alloc()                     <- nothing writes it
    linalg.generic ins(%alloc_2, %alloc_3) outs(%alloc_2)

so the second gather's index loses its `x0 + M*x2` term, reads whatever the
scratchpad held, and the DMA walks off the end of DRAM:

    User load segfault @ 0x00000000d804d820      [PyTorchSim tests/ops/attention/test_gqa.py]

The same guard already exists three lines above for the WRITER being moved --
"a write that also reads what it writes cannot move" -- and this is that
sentence applied to the other end.

WHY THE COUNT AND NOT THE VERDICT. Whether the rewrite fires at all depends on
ASLR: the same tree, the same input, compiled ten times, answers two different
IRs. Measured -- `setarch -R` gives 6 of 6 identical, without it 8 runs of this
kernel failed twice. A pass/fail witness would therefore be a coin flip, so the
claim is the SNAPSHOT ITSELF: six `torchsim.vector_loop` in the correct
lowering, five when the copy has been taken away. None of the 477 kernels in
this suite flips (measured, two processes each), which is why nothing here
caught it.

AND IT COUNTS THE VECTOR LOOP, NOT `affine.for`, which used to say the same six.
A vector loop is one per buffer written, so it is the loop a snapshot IS; a bare
`affine.for` is however many loops the nest happens to have. Under the bank view
a buffer's type carries a depth dim ahead of the lane, and vectorize walks
dims, so every one of the six nests gains an outer `affine.for %d = 0 to 1` and
the bare count is 12. Six vector loops on either path, and five is still what
says a copy has gone.

WHY A PROGRAM ID. With `grid=(1,)` the index tile is a constant `tl.arange` and
constant-folds; nothing is computed into a shared buffer and the kernel passes
on either tree. The dynamic `xoffset` is what makes `xindex` a computed tile.

torch equivalent: (a[x0 + M*x2] + b[x0 + M*x2 + S*x1]) * 0.3194715.

Two gathers sharing an index buffer: pins the snapshot copy that bufferization makes and _write_where_the_copy_lands must not remove.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

XB, M, N, S = 256, 96, 3072, 1536
XNUMEL = 24576


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def inductor_gather_index_snapshot_kernel(in_ptr0, in_ptr1, out_ptr0,
                                              xnumel, XB: tl.constexpr,
                                              M: tl.constexpr, N: tl.constexpr,
                                              S: tl.constexpr):
        xoffset = tl.program_id(0) * XB
        xindex = xoffset + tl.arange(0, XB)
        x0 = xindex % M
        x2 = xindex // N
        x1 = (xindex // M) % 32
        a = tl.load(in_ptr0 + (x0 + M * x2))
        b = tl.load(in_ptr1 + (x0 + M * x2 + S * x1))
        tl.store(out_ptr0 + xindex, (a + b) * 0.3194715)

    return inductor_gather_index_snapshot_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"in_ptr0": torch.randn(1536, generator=g, dtype=torch.float32),
            "in_ptr1": torch.randn(49152, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    i = torch.arange(XNUMEL)
    x0, x2, x1 = i % M, i // N, (i // M) % 32
    a = inputs["in_ptr0"][x0 + M * x2]
    b = inputs["in_ptr1"][x0 + M * x2 + S * x1]
    return {"out_ptr0": (a + b) * 0.3194715}


SPEC = KernelSpec(
    name="inductor_gather_index_snapshot_kernel", kernel=kernel,
    signature={"in_ptr0": "*fp32", "in_ptr1": "*fp32", "out_ptr0": "*fp32",
               "xnumel": "i32", "XB": "constexpr", "M": "constexpr",
               "N": "constexpr", "S": "constexpr"},
    constexprs={"XB": XB, "M": M, "N": N, "S": S},
    args=[Arg("in_ptr0", IN, "float32", (1536,)),
          Arg("in_ptr1", IN, "float32", (49152,)),
          Arg("out_ptr0", OUT, "float32", (XNUMEL,))],
    grid=(96, 1, 1), parallel_axes=('x',),
    extra={"scalar_args": [('xnumel', 'int32_t')],
           "scalar_values": {'xnumel': XNUMEL}},
    reference=reference, make_inputs=make_inputs,
    # THE SNAPSHOT IS THE CLAIM, not the verdict: whether the rewrite that takes
    # it away fires is decided by ASLR, so pass/fail alone is a coin flip.
    expect_ir={"04-passes/07-torchsim-vectorize.mlir": {"torchsim.vector_loop": 6}},
)
