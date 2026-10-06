"""A TUPLE `tl.associative_scan` -- one `tt.scan` carrying two values, which is
the shape Mamba/SSM selective scans are written in.

WHAT IT PINS. `tt.scan` is `Variadic<TT_Tensor>:$srcs` -> `Variadic<TT_Tensor>`
(TritonOps.td:781-782) and `tt.scan.return` is variadic to match, so a tuple
scan is ONE op with two operands, two results and a four-argument combiner
region -- the left tuple first, then the right:

    ^bb0(%a1: f32, %a2: f32, %b1: f32, %b2: f32):
      %11 = arith.addf %a1, %b1
      %12 = arith.mulf %a2, %b2
      tt.scan.return %11, %12

_unroll_scan runs ONE LADDER PER VALUE over one padded buffer each, and each
step is a single multi-result linalg.generic so the combiner is evaluated once
however coupled its two halves are.

WHAT IT FAILED WITH BEFORE. The ladder read index 0 in three places -- the
operands, the results and the terminator -- so the op was erased with its
second result still used, and the canonicalize at the end of the pass walked
the dangling value:

    KILLED by signal 11 (SIGSEGV) in p00_convert_unsupported_ops.py run

WHAT WOULD BREAK IT. Counting OPERANDS to find how many values a scan carries.
`ttx.cumsum` is destination-passing -- `Variadic<AnyType>:$inputs,
Variadic<AnyType>:$outputs` (TritonTilingExtOps.td:205-210) -- so a one-value
cumsum already has two operands, and an operand count refuses seven passing
kernels in this suite. The count is `len(op.results)`.

A two-value tt.scan, and the first multi-result linalg.generic this compiler emits. The two halves are independent here so a wrong pairing of ladder to buffer shows up as a swapped output rather than as a tolerance miss.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 32


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _both(a1, a2, b1, b2):
        return a1 + b1, a2 * b2

    @triton.jit
    def ops_scan_two_values_kernel(x_ptr, y_ptr, s_ptr, p_ptr,
                                   N: tl.constexpr):
        r = tl.arange(0, N)
        x = tl.load(x_ptr + r)
        y = tl.load(y_ptr + r)
        s, p = tl.associative_scan((x, y), 0, _both)
        tl.store(s_ptr + r, s)
        tl.store(p_ptr + r, p)

    return ops_scan_two_values_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(N, generator=g, dtype=torch.float32),
            "y": torch.rand(N, generator=g, dtype=torch.float32) * 0.5 + 0.9}


def reference(inputs):
    return {"s": inputs["x"].cumsum(dim=0), "p": inputs["y"].cumprod(dim=0)}


SPEC = KernelSpec(
    name="ops_scan_two_values_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "s_ptr": "*fp32",
               "p_ptr": "*fp32", "N": "constexpr"},
    constexprs={"N": N},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", IN, "float32", (N,)),
          Arg("s", OUT, "float32", (N,)), Arg("p", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
