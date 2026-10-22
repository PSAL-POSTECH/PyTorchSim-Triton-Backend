"""erfcinv/normcdfinv below 2^-24: w comes from y directly, not through 1-y --
before, erfcinv(1e-8) and normcdfinv(1e-8) both returned inf (true 4.05, -5.61).
Breaks it: composing onto erfinv again, or dropping normcdfinv's -sqrt2 sign.

convert_unsupported_ops, POSITIVE. erfcinv(y) = erfinv(1-y) and normcdfinv(p) = sqrt2*erfinv(2p-1) both reintroduce the cancellation _derive_erfc_from_erf_and_tail exists to avoid: in f32, 1-y rounds to 1.0 for every y below 2^-24, so erfinv w = -log(1 - x*x) becomes -log(0) = inf. MEASURED BEFORE on this kernel: erfcinv(1e-8) and normcdfinv(1e-8) both inf, true 4.0522 and -5.6120. AFTER: PASS, max_rel_err 8.35e-06 and 2.46e-06. The composition threw away an exact algebraic form: erfinv argument enters only through w = -log((1-x)(1+x)), and for x = 1-y that is -log(y*(2-y)) -- the same expression with no subtraction of near- equal numbers. So the polynomial half is factored into _derive_erfinv_from_w and each entry point computes its own w; normcdfinv routes through erfcinv as Phi^-1(p) = -sqrt2*erfcinv(2p). STILL LOST, stated rather than claimed: the UPPER tail (y near 2, p near 1) cancels in 2-y, and no single f32 formula holds both tails without a branch. The inputs carry a decade ladder 1e-8..1e-1 into the lower tail. NOTE: this body holds two erfinv evaluations and legalize once warned "51 of 32" live values at the old MIN_VEC of 8; it runs and verifies, but it is the closest kernel in this family to the register file.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
BLOCK = 128


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_inverse_tails_kernel(y_ptr, p_ptr, a_ptr, b_ptr,
                                      BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        y = tl.load(y_ptr + r)
        p = tl.load(p_ptr + r)
        tl.store(a_ptr + r, libdevice.erfcinv(y))
        tl.store(b_ptr + r, libdevice.normcdfinv(p))

    return ops_math_inverse_tails_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    tail = torch.tensor([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                        dtype=torch.float32)
    y = torch.rand(N, generator=g, dtype=torch.float32) * 1.6 + 0.2
    y[:tail.numel()] = tail
    p = torch.rand(N, generator=g, dtype=torch.float32) * 0.98 + 0.01
    p[:tail.numel()] = tail
    return {"y": y.contiguous(), "p": p.contiguous()}


def reference(inputs):
    import torch
    y, p = inputs["y"].to(torch.float64), inputs["p"].to(torch.float64)
    a = torch.erfinv(1.0 - y)
    b = torch.erfinv(2.0 * p - 1.0) * (2.0 ** 0.5)
    return {"a": a.to(torch.float32).contiguous(),
            "b": b.to(torch.float32).contiguous()}


SPEC = KernelSpec(
    name="ops_math_inverse_tails_kernel",
    kernel=kernel,
    signature={"y_ptr": "*fp32", "p_ptr": "*fp32", "a_ptr": "*fp32",
               "b_ptr": "*fp32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("y", IN, "float32", (N,)), Arg("p", IN, "float32", (N,)),
          Arg("a", OUT, "float32", (N,)), Arg("b", OUT, "float32", (N,))],
    grid=(1,),
    reference=reference,
    make_inputs=make_inputs,
    rtol=3e-3, atol=3e-4,
)
