"""`tl.exp` and `tl.log` of a SCALAR read from DRAM -- no reduction anywhere.

WHAT IT PINS, AND WHY IT IS NOT A COPY OF reduce_scalar_exp.py. That kernel's
scalar is made by folding a tile to nothing, so a reader can believe the answer
lies in how a reduction ends. This one has no reduction at all: `tl.load(g_ptr +
i)` on a plain pointer is a `tts.gather` returning an f32, and there was never a
tile for the value to come from. The two are the suite's two SOURCES of a scalar
transcendental (they are gated_delta_rule's and flash_decode_splitk's
respectively), and one fix has to answer both -- if only one of these passes,
whatever made it pass was a rule about reductions and not about scalars.

The gate is built the way a delta-rule model builds it, `exp(-softplus(g))`, so
the chain is exp -> log -> exp on a scalar and NOT a single op: an f32 result
feeds an f32 operand two ops later, which is where a rewrite that turns each op
into a tile and forgets to come back out shows up as a verifier failure rather
than as a wrong number.

WITHOUT scalar_math_to_tile this dies in stage `lower` with

    math.exp on a SCALAR has nowhere to go: VCIX takes a vector register

Verified against the tree before that pass.

WHAT WOULD BREAK IT. A rewrite that only looks at reductions. Also a rewrite
that lifts the whole scalar chain into the vector domain and comes back with
`vector.extract`: a vector made scalar, which c27_verify_lane_values_stay_vector
rejects in stage `lower`, so the failure would move rather than go.

A scalar DRAM read feeding exp/log/exp. No reduction, so a fix aimed at how reductions end does not reach it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_scalar_math_load_kernel(x_ptr, g_ptr, o_ptr, N: tl.constexpr):
        rn = tl.arange(0, N)
        x = tl.load(x_ptr + rn)
        g = tl.load(g_ptr + 0)                       # ONE element: an f32
        gate = tl.exp(0.0 - tl.log(1.0 + tl.exp(g)))  # exp(-softplus(g))
        tl.store(o_ptr + rn, x * gate)

    return ops_scalar_math_load_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "g": torch.randn(1, generator=g, dtype=torch.float32) + 2.0}


def reference(inputs):
    import torch
    gate = torch.exp(-torch.nn.functional.softplus(inputs["g"][0]))
    return {"o": inputs["x"] * gate}


SPEC = KernelSpec(
    name="ops_scalar_math_load_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "g_ptr": "*fp32", "o_ptr": "*fp32",
               "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("g", IN, "float32", (1,)),
          Arg("o", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
