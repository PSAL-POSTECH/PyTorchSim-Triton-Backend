"""Index tables over more than one axis: one now supported, the others still refused.

A value that varies on several axes is ONE index over those axes (a joint state
in PtrAnalysis): tts.make_gather_scatter_tptr lists them in gather_scatter_dims
and the index stands at the tile's rank. What is still refused is an address
that would need two index tables, or a remainder of one.

NOT A FAILING KERNEL. The refused loads are picked up by
--triton-to-unstructured as plain tts.gather ops, so the pipeline succeeds end
to end. "Refused" here means "no tts.make_gather_scatter_tptr was built", not
"the compile broke".

  load 1   SUPPORTED. The loaded index is itself 8x2, so both axes are
           gathered: rebuildAsUnsupportedOp makes a joint state and the pointer
           is
               tts.make_gather_scatter_tptr %arg0 to sizes: [8, 2]
                   gather_scatter_dims = [0, 1], offset tensor<8x2xi32>
           Every axis is gathered, so each row is one element and
           convert-accesses-to-transfer reads it in ONE element-wise indirect
           transfer -- the same DMA the tts.gather gave, from a gather pointer.
  load 2   REFUSED, rebuildAsGatherScatter: `(i8[:, None]*16 + arange(16)[None,
           :]) % 64` is an 8x16 remainder of a gather; the modulo rebuild has
           one gathered axis to put it on and there are two.
  load 3   REFUSED, addState: two independent loaded index tensors, one on each
           axis (`i8[:, None]*16 + j16[None, :]`). Each side is a one-axis
           gather on a different axis, and their sum is two index tables.

WHAT PROVES IT
  06-linalg.mlir holds one tts.make_gather_scatter_tptr (load 1, with
  gather_scatter_dims = array<i32: 0, 1>) and two tts.gather (loads 2 and 3),
  each with its full 8x16 offset tensor. The two STORES are unaffected and stay
  structured tts.store.

torch equivalent: compile-time probe (`reference` returns zeros); the file
exists for the refusals, and load 2's modulo makes the addresses uninteresting
numerically.

Three shapes with more than one non-contiguous dim: the loaded 2-D index is one gather pointer over both axes, the remainder of a gather and the sum of two tables are refused and recovered as element-wise tts.gather.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

NX = 4096
NI = 1024
M = 64
BM = 8
BN = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_gather_multidim_refused_kernel(x_ptr, i_ptr, o_ptr, o2_ptr,
                                           M: tl.constexpr, BM: tl.constexpr,
                                           BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        r2 = tl.arange(0, 2)
        i8 = tl.load(i_ptr + rm)                 # 8 indices, axis 0
        j16 = tl.load(i_ptr + 16 + rn)           # 16 indices, axis 1

        # 1. the loaded index tensor is itself 8x2 -- two dims of extent > 1
        i2d = tl.load(i_ptr + rm[:, None] * 2 + r2[None, :])
        a = tl.load(x_ptr + i2d)
        # 2. a modulo whose operand is an 8x16 gather expression
        b = tl.load(x_ptr + ((i8[:, None] * BN + rn[None, :]) % M))
        # 3. two independent gathers, one per axis
        c = tl.load(x_ptr + i8[:, None] * BN + j16[None, :])

        tl.store(o_ptr + rm[:, None] * 2 + r2[None, :], a)
        tl.store(o2_ptr + rm[:, None] * BN + rn[None, :], b + c)

    return ptr_gather_multidim_refused_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(NX, generator=g, dtype=torch.float32),
            "i": torch.randint(0, M, (NI,), generator=g, dtype=torch.int32)}


def reference(inputs):
    """The three gathers, as written.

    This was zeros() -- a placeholder from when the file could not be compiled
    and the numbers were never reached. PtrAnalysis declining these shapes is
    what the file is here for; it still lowers, through tts.gather.
    """
    import torch
    x, i = inputs["x"], inputs["i"].long()
    rm, rn, r2 = torch.arange(BM), torch.arange(BN), torch.arange(2)
    i8, j16 = i[:BM], i[16:16 + BN]
    a = x[i[rm[:, None] * 2 + r2[None, :]]]                    # 2-D index tensor
    b = x[(i8[:, None] * BN + rn[None, :]) % M]                # modulo of a gather
    c = x[i8[:, None] * BN + j16[None, :]]                     # one gather per axis
    return {"o": a.reshape(-1), "o2": (b + c).reshape(-1)}


SPEC = KernelSpec(
    name="ptr_gather_multidim_refused_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "i_ptr": "*i32", "o_ptr": "*fp32",
               "o2_ptr": "*fp32", "M": "constexpr", "BM": "constexpr",
               "BN": "constexpr"},
    constexprs={"M": M, "BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (NX,)), Arg("i", IN, "int32", (NI,)),
          Arg("o", OUT, "float32", (BM * 2,)),
          Arg("o2", OUT, "float32", (BM * BN,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    # load 1 is a gather pointer over both axes; loads 2 and 3 stay per element.
    expect_ir={"07-regularized.mlir": {"tts.make_gather_scatter_tptr": 1,
                                       "tts.gather": 2}},
)
