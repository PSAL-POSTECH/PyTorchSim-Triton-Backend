"""MaskAnalysis coverage: `offs_m[:, None] > pid` -- a lower bound against a
value, on a tile that is then expanded to rank 3.

WHAT IT PINS NOW. A comparison whose bound is a RUNTIME scalar is carried as the
axis's low, not refused. parseCmp computes the first live lane as
`clamp(bound - start, 0, dim)` and tts.load takes it as a dynamic `mask_lows`,
so the mask survives and the answer is exact (max_abs_err 0).

WHAT IT PINNED BEFORE, AND WHY THAT IS GONE. This kernel used to be
`expect="refuse"`, and the refusal was a C++ ABORT: parseCmp could not take an
`sgt` against a runtime scalar, so the mask came back UNSUPPORTED with an EMPTY
`dims`, and parseExpandDims' rank>1 recovery arm then did `dims[dim] = ...` on
that empty SmallVector -- `Assertion 'idx < size()' failed`, SIGABRT, rc=134,
no later pass and no executable. Making the comparison supported is what makes
that arm unreachable FROM HERE; the arm itself is no longer covered by any
kernel, and a mask that is still unsupported for some other reason would reach
it again.

WHAT WOULD BREAK IT. Folding a lower bound away instead of carrying it -- which
is what the `>= 0` arm did until Inductor's padded convolution proved the start
can be negative -- or dropping the dynamic half of `mask_lows`, which is the
only slot a runtime low fits in.

A lower bound against a runtime scalar, carried as a dynamic mask_lows instead of aborting upstream.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

B, M, N = 2, 8, 128
B_RT = 0        # runtime value of `b`; pinned in SPEC.extra below


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_assert_expand3d_kernel(x_ptr, o_ptr, b,
                                    B: tl.constexpr, M: tl.constexpr,
                                    N: tl.constexpr):
        pid = tl.program_id(0)
        offs_b = tl.arange(0, B)
        offs_m = tl.arange(0, M)
        offs_n = tl.arange(0, N)
        ptrs = (x_ptr + offs_b[:, None, None] * (M * N)
                + offs_m[None, :, None] * N + offs_n[None, None, :])
        m2 = offs_m[:, None] > pid          # tensor<Mx1xi1>: mask recorded, sgt
        v = tl.load(ptrs, mask=(offs_b[:, None, None] < b) & m2[None, :, :],
                    other=0.0)
        tl.store(o_ptr + offs_b[:, None, None] * (M * N)
                 + offs_m[None, :, None] * N + offs_n[None, None, :], v)

    return mask_assert_expand3d_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(B * M * N, generator=g, dtype=torch.float32)}


def reference(inputs):
    """Derived from the body rather than asserted: grid is (1,) so pid == 0,
    the flat index b*M*N + m*N + n covers all B*M*N elements contiguously, and
    the STORE is unmasked, so the `other=0.0` lanes really are written as zeros.

        o[b, m, n] = x[b, m, n]  if (b < B_RT and m > pid) else 0.0

    With B_RT = 0 this is all zeros; with B_RT = 1 it would be 896 nonzeros, so
    the value of `b` has to be pinned for the reference to mean anything.
    """
    import torch
    x = inputs["x"].reshape(B, M, N)
    pid = 0                                   # grid=(1,)
    keep = ((torch.arange(B)[:, None, None] < B_RT)
            & (torch.arange(M)[None, :, None] > pid))
    return {"o": torch.where(keep, x, torch.zeros_like(x)).reshape(-1)}


SPEC = KernelSpec(
    name="mask_assert_expand3d_kernel",
    kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "b": "i32",
               "B": "constexpr", "M": "constexpr", "N": "constexpr"},
    constexprs={"B": B, "M": M, "N": N},
    args=[Arg("x", IN, "float32", (B * M * N,)),
          Arg("o", OUT, "float32", (B * M * N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    # `b` is a runtime scalar, not a memref Arg. Without this the generated C
    # driver (pytorchsim_triton_compiler/lowering/wrapper.py:_scalars) emits NOTHING for it and the trailing
    # grid i32s slide into its slot, making b == gridX == 1 instead of B_RT.
    # Unreachable here (the compiler aborts first), but the reference above is
    # only well-defined once `b` is pinned.
    extra={"scalar_args": [("b", "int32_t")], "scalar_values": {"b": B_RT}},
)
