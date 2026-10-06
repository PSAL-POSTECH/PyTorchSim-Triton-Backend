"""A masked load through a tensor of pointers, `other=` a runtime scalar: one tts.gather with mask and fill.
Breaks if convert-unsupported-pointers stops reading the scf.if the access arrives in, or the uniform fill."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
LIVE = 100
FILL = -3.5


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ptr_masked_gather_uniform_other_kernel(a_ptr, b_ptr, o_ptr, fill,
                                               LIVE: tl.constexpr,
                                               BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        p = tl.where((r % 2) == 0, a_ptr + r, b_ptr + r)
        tl.store(o_ptr + r, tl.load(p, mask=r < LIVE, other=fill))

    return ptr_masked_gather_uniform_other_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(N, generator=g, dtype=torch.float32),
            "b": torch.randn(N, generator=g, dtype=torch.float32)}


def reference(inputs):
    import torch
    r = torch.arange(N)
    v = torch.where(r % 2 == 0, inputs["a"], inputs["b"])
    return {"o": torch.where(r < LIVE, v, torch.full_like(v, FILL))}


SPEC = KernelSpec(
    name="ptr_masked_gather_uniform_other_kernel", kernel=kernel,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "o_ptr": "*fp32",
               "fill": "fp32", "LIVE": "constexpr", "BLOCK": "constexpr"},
    constexprs={"LIVE": LIVE, "BLOCK": N},
    args=[Arg("a", IN, "float32", (N,)), Arg("b", IN, "float32", (N,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    extra={"scalar_args": [("fill", "float")],
           "scalar_values": {"fill": FILL}},
)
