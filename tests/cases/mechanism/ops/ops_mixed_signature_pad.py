"""Two linalg.generics with DIFFERENT signatures in one func, both needing a pad.

An f32 chain and an i32 chain side by side, at dim0 = 8 so the lane-multiple pad
actually fires on each. This is the exact shape `fuse_and_pad` could not express
while it padded in bulk: one `transform.structured.match ops{["linalg.generic"]}`
covering every generic needs ONE padding_values list, and padding_values is
per-OPERAND of the matched op, so every generic in the func had to agree on
operand count, element types and rank.

Verified against the pre-change pass on this kernel's own stage-2 output:

    OLD: _Fatal: generics with differing signatures in one func
         ((2, ('f32','f32'), 2) vs (2, ('i32','i32'), 2))
    NEW: Spike, max_abs_err = 0 on both outputs

The restriction was an artefact of matching in bulk, not a real one -- a per-op
sequence gives each generic its own padding_values. Same for linalg.fill, whose
(scalar, tensor) operand shape never matched a generic's, which is why the pass
used to carry two separate transform templates.

WHY dim0 = 8 AND NOT A LANE MULTIPLE
  If dim0 were already 128 nothing would be padded and the templates would never
  be built, so the kernel would pass whether or not the restriction existed. The
  small tile is what makes this a test.

torch equivalent: o = x*2+1 (f32), p = y*3-5 (i32).

Mixed-signature generics that both get padded -- the case bulk matching could not express. Also a small-tile (dim0 < lanes) case.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

BM, BN = 8, 16
N = BM * BN


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def ops_mixed_signature_pad_kernel(x_ptr, y_ptr, o_ptr, p_ptr,
                                       BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        off = rm[:, None] * BN + rn[None, :]
        a = tl.load(x_ptr + off)          # f32 chain
        b = tl.load(y_ptr + off)          # i32 chain
        tl.store(o_ptr + off, a * 2.0 + 1.0)
        tl.store(p_ptr + off, b * 3 - 5)

    return ops_mixed_signature_pad_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "y": torch.randint(-50, 50, (N,), generator=g, dtype=torch.int32)}


def reference(inputs):
    return {"o": inputs["x"] * 2.0 + 1.0, "p": inputs["y"] * 3 - 5}


SPEC = KernelSpec(
    name="ops_mixed_signature_pad_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*i32", "o_ptr": "*fp32",
               "p_ptr": "*i32", "BM": "constexpr", "BN": "constexpr"},
    constexprs={"BM": BM, "BN": BN},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", IN, "int32", (N,)),
          Arg("o", OUT, "float32", (N,)), Arg("p", OUT, "int32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
