"""Two gathers whose index tiles share a buffer, the second computed FROM the first in place -- and the
snapshot copy between them that must survive."""
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
    expect_ir={"08-passes/07-pytorchsim-vectorize.mlir": {"torchsim.vector_loop": 5}},
)
