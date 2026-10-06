"""The mirror of ptr_wrap_meets_gather: gather on axis 0, wrap on axis 1, and
the modulus is a RUNTIME argument.

    idx = j[r] * C + (c % M)          gather on axis 0, wrap on axis 1

Two things this pins that the constant-modulus twin cannot. The wrap slot takes
a dynamic extent, so it is an operand and not just a static_shape entry; and the
axis it lands on is the LAST one, where the descriptor's own walk is the unit
stride -- so the fold into the index has to replace that walk, not ride beside
it.

M=5 against C=8, so the wrap fires on columns 5..7.

A runtime wrap extent on the last axis of a gather pointer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, M = 8, 8, 5
ASIZE = 64


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wrap_gather_last_axis(a_ptr, j_ptr, m_ptr, o_ptr, R: tl.constexpr,
                              C: tl.constexpr):
        r = tl.arange(0, R)
        c = tl.arange(0, C)
        j = tl.load(j_ptr + r)
        m = tl.load(m_ptr)                       # the modulus, at runtime
        v = tl.load(a_ptr + j[:, None] * C + (c % m)[None, :])
        tl.store(o_ptr + (r[:, None] * C + c[None, :]), v)

    return wrap_gather_last_axis


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(ASIZE, generator=g, dtype=torch.float32),
            "j": torch.randint(0, R, (R,), generator=g, dtype=torch.int32),
            "m": torch.tensor([M], dtype=torch.int32)}


def reference(inputs):
    import torch
    a, j = inputs["a"], inputs["j"].long()
    c = torch.arange(C)
    return {"o": a[j[:, None] * C + (c % M)[None, :]].reshape(-1).contiguous()}


SPEC = KernelSpec(
    name="wrap_gather_last_axis", kernel=kernel,
    signature={"a_ptr": "*fp32", "j_ptr": "*i32", "m_ptr": "*i32",
               "o_ptr": "*fp32", "R": "constexpr", "C": "constexpr"},
    constexprs={"R": R, "C": C},
    args=[Arg("a", IN, "float32", (ASIZE,)),
          Arg("j", IN, "int32", (R,)),
          Arg("m", IN, "int32", (1,)),
          Arg("o", OUT, "float32", (R * C,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-4,
    # The ONE tts.gather is the scalar `tl.load(m_ptr)`, not the tile: a
    # scalar offset is no transfer and has nowhere else to go.
    expect_ir={"03-ttshared.mlir": {"tts.make_gather_scatter_tptr": 1,
                                    "tts.gather": 1}},
)
