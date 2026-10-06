"""The five libdevice symbols that needed a NAME, not a derivation.

_EXTERN_NATIVE maps them onto ops the vector unit already has:

    __nv_copysignf  -> math.copysign      __nv_rintf       -> math.roundeven
    __nv_fmaf       -> math.fma           __nv_nearbyintf  -> math.roundeven
    __nv_roundf     -> math.round

and until this file the table had five entries and NO kernel -- so every one of
them was an untested arm of convert_unsupported_ops, and the five NATIVE_MATH ops they
choose between were only ever reached through the math.* door.

WHY THEY REACH US AT ALL. triton-shared's ExternElementwise converters cover 52
symbols, 48 unary and 4 binary, and none of these five is among them, so the
tt.extern_elementwise survives stage 2 with no body. That is upstream's table
being short rather than anything about this machine (rule 11) -- three of the
five are one unary row, copysign is one binary row, and only fma needs a
converter upstream does not have at all, being ternary.

WHAT WOULD BREAK IT. Dropping any row of _EXTERN_NATIVE: the symbol then falls
to _choose_canon_of_symbol's `None` arm and convert_unsupported_ops refuses by name.
Routing one of them through _DERIVE instead: rint and nearbyint would still
compute the right number, and copysign/fma/round would each cost a derivation
where a single instruction exists.

rint and nearbyint are the same op on purpose -- both are round-to-nearest-even
and differ only in whether the inexact flag is raised, which nothing here
models.

torch equivalent: copysign, addcmul, and three roundings of the same tile.

The five _EXTERN_NATIVE symbols, which had no kernel. Pins that a libdevice symbol upstream does not convert still reaches a single native op rather than a derivation.
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
    def ops_math_extern_native_kernel(x_ptr, y_ptr, o0, o1, o2, o3, o4,
                                      BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + r)
        y = tl.load(y_ptr + r)
        tl.store(o0 + r, libdevice.copysign(x, y))
        tl.store(o1 + r, libdevice.fma(x, y, x))
        tl.store(o2 + r, libdevice.rint(x))
        tl.store(o3 + r, libdevice.nearbyint(x))
        tl.store(o4 + r, libdevice.round(x))

    return ops_math_extern_native_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Spread across .5 boundaries so round (half away from zero) and roundeven
    # (half to even) disagree on some elements, and across both signs so
    # copysign has something to carry.
    x = (torch.rand(N, generator=g, dtype=torch.float32) - 0.5) * 12.0
    x[:8] = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5, -3.5])
    y = (torch.rand(N, generator=g, dtype=torch.float32) - 0.5) * 4.0
    return {"x": x, "y": y}


def reference(inputs):
    import torch
    x, y = inputs["x"], inputs["y"]
    half = torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))
    return {"o0": torch.copysign(x, y),
            "o1": x * y + x,
            "o2": torch.round(x),          # round-half-to-even
            "o3": torch.round(x),
            "o4": half}                    # round-half-away-from-zero


SPEC = KernelSpec(
    name="ops_math_extern_native_kernel", kernel=kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "o0": "*fp32",
               "o1": "*fp32", "o2": "*fp32", "o3": "*fp32", "o4": "*fp32",
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": BLOCK},
    args=[Arg("x", IN, "float32", (N,)), Arg("y", IN, "float32", (N,)),
          Arg("o0", OUT, "float32", (N,)), Arg("o1", OUT, "float32", (N,)),
          Arg("o2", OUT, "float32", (N,)), Arg("o3", OUT, "float32", (N,)),
          Arg("o4", OUT, "float32", (N,))],
    grid=(1,), reference=reference, make_inputs=make_inputs,
)
