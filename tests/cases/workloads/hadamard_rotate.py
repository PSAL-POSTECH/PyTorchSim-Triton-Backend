"""QuaRot / SpinQuant `rotate_activation` -- a Hadamard transform over the feature dim, done as a log-depth
butterfly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

T, D = 16, 16          # tokens, feature dim -- log2(16) = 4 butterfly stages
INV = 0.25             # 1/sqrt(D); a power of two, so exact in fp32


def kernel():
    import triton
    import triton.language as tl

    # ONE BUTTERFLY STAGE. `g`, `h` and the scale are constexpr PARAMETERS, so
    # the recursion's geometry is in the signature and the caller supplies
    # literals -- see the module docstring for why nothing else works.
    @triton.jit
    def _bfly(x, T: tl.constexpr, g: tl.constexpr, h: tl.constexpr,
              sc: tl.constexpr):
        # [T, D] -> [T, g, 2, h]: bit `h` of the flat index becomes axis 2.
        y = tl.reshape(x, [T, g, 2, h])
        y = tl.permute(y, (0, 1, 3, 2))          # partners to the last axis
        lo, hi = tl.split(y)                     # index bit clear / set
        y = tl.join((lo + hi) * sc, (lo - hi) * sc)
        y = tl.permute(y, (0, 1, 3, 2))
        return tl.reshape(y, [T, g * 2 * h])

    @triton.jit
    def hadamard_rotate_kernel(x_ptr, o_ptr, T: tl.constexpr, D: tl.constexpr,
                               INV: tl.constexpr):
        rt = tl.arange(0, T)
        rd = tl.arange(0, D)
        off = rt[:, None] * D + rd[None, :]

        x = tl.load(x_ptr + off)
        # The distance between partners DOUBLES every stage: 1, 2, 4, 8.
        x = _bfly(x, T, 8, 1, 1.0)
        x = _bfly(x, T, 4, 2, 1.0)
        x = _bfly(x, T, 2, 4, 1.0)
        x = _bfly(x, T, 1, 8, INV)      # 1/sqrt(D) folded into the last stage
        tl.store(o_ptr + off, x)

    return hadamard_rotate_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # An activation with the per-channel outliers the rotation exists to kill:
    # two feature dims at 30x, identically in every token.
    x = torch.randn(T, D, generator=g, dtype=torch.float32)
    heavy = torch.zeros(1, D)
    heavy[0, 3] = heavy[0, 11] = 1.0
    return {"x": x * (1.0 + 29.0 * heavy)}


def reference(inputs):
    import torch
    # The same four stages in the same order, so this is bit-identical to the
    # kernel rather than an approximation of it.
    x = inputs["x"].reshape(T, D).clone()
    h = 1
    while h < D:
        y = x.reshape(T, D // (2 * h), 2, h)
        lo, hi = y[:, :, 0, :], y[:, :, 1, :]
        sc = INV if 2 * h == D else 1.0
        x = torch.stack([(lo + hi) * sc, (lo - hi) * sc], dim=2).reshape(T, D)
        h *= 2
    return {"o": x.contiguous()}


SPEC = KernelSpec(
    name="hadamard_rotate_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "o_ptr": "*fp32", "T": "constexpr",
               "D": "constexpr", "INV": "constexpr"},
    constexprs={"T": T, "D": D, "INV": INV},
    args=[Arg("x", IN, "float32", (T, D)),
          Arg("o", OUT, "float32", (T, D))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-5, atol=1e-5,
    # The butterfly's half written back into its tile is ONE permute that writes the
    # window's rows and lanes and leaves the rest -- no DRAM trip. Stage 07 takes the split's
    # unit-axis slices rank-reduced, each with its layout: no reshape pair or tensor.cast around them.
    expect_ir={"08-passes/04-pytorchsim-materialize-layouts.mlir": {"pytorchsim.convert_layout": 0},
               "07-regularized.mlir": {"tensor.collapse_shape": 4, "tensor.cast": 0}},
)
