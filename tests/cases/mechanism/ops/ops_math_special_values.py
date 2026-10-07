"""Special values of the derived math: asin/acos past |x| = 1, the erfinv family's far tail and
endpoints, lgamma's and tgamma's poles, and pow's C cases."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = 128
NAN, INF = float("nan"), float("inf")


def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    globals()["libdevice"] = libdevice

    @triton.jit
    def ops_math_special_values_kernel(u_ptr, y_ptr, v_ptr, p_ptr, g_ptr, x_ptr, z_ptr,
                                       as_ptr, ac_ptr, ec_ptr, ei_ptr, nc_ptr, lg_ptr, tg_ptr,
                                       pw_ptr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        u = tl.load(u_ptr + r)
        g = tl.load(g_ptr + r)
        tl.store(as_ptr + r, libdevice.asin(u))
        tl.store(ac_ptr + r, libdevice.acos(u))
        tl.store(ec_ptr + r, libdevice.erfcinv(tl.load(y_ptr + r)))
        tl.store(ei_ptr + r, libdevice.erfinv(tl.load(v_ptr + r)))
        tl.store(nc_ptr + r, libdevice.normcdfinv(tl.load(p_ptr + r)))
        tl.store(lg_ptr + r, libdevice.lgamma(g))
        tl.store(tg_ptr + r, libdevice.tgamma(g))
        tl.store(pw_ptr + r, libdevice.pow(tl.load(x_ptr + r), tl.load(z_ptr + r)))

    return ops_math_special_values_kernel


def _with(torch, head, rest):
    t = rest.clone()
    t[:len(head)] = torch.tensor(head, dtype=torch.float32)
    return t


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda lo, hi: torch.rand(N, generator=g, dtype=torch.float32) * (hi - lo) + lo
    return {
        "u": _with(torch, [1.5, -1.5, 2.0, 1.0, -1.0], rnd(-0.9, 0.9)),
        "y": _with(torch, [1e-8, 1e-10, 1e-16, 1e-22, 1e-30, 0.0, 2.0], rnd(0.2, 1.8)),
        "v": _with(torch, [1.0, -1.0], rnd(-0.9, 0.9)),
        "p": _with(torch, [0.0, 1.0, 1e-20], rnd(0.01, 0.99)),
        "g": _with(torch, [-1.0, -2.0, 0.0, -0.0], rnd(0.6, 5.0)),
        "x": _with(torch, [1.0, 1.0, -1.0, -1.0, 0.0, 2.0, NAN], rnd(0.5, 3.0)),
        "z": _with(torch, [NAN, INF, INF, -INF, NAN, NAN, 0.0], rnd(-2.0, 2.0)),
    }


def _tgamma(torch, g):
    import math
    out = []
    for v in g.tolist():
        if v == 0.0:
            out.append(math.copysign(INF, v))
        elif v < 0 and v == int(v):
            out.append(NAN)
        else:
            out.append(math.gamma(v))
    return torch.tensor(out, dtype=torch.float64)


def reference(inputs):
    import math
    import torch
    d = {k: v.double() for k, v in inputs.items()}
    out = {
        "as_": torch.asin(d["u"]), "ac": torch.acos(d["u"]),
        "ec": -torch.special.ndtri(d["y"] / 2) / math.sqrt(2.0),
        "ei": torch.special.erfinv(d["v"]), "nc": torch.special.ndtri(d["p"]),
        "lg": torch.lgamma(d["g"]), "tg": _tgamma(torch, d["g"]),
        "pw": torch.pow(d["x"], d["z"]),
    }
    return {k: v.float() for k, v in out.items()}


SPEC = KernelSpec(
    name="ops_math_special_values_kernel", kernel=kernel,
    signature={**{f"{a}_ptr": "*fp32" for a in ("u", "y", "v", "p", "g", "x", "z", "as",
                                                 "ac", "ec", "ei", "nc", "lg", "tg", "pw")},
               "BLOCK": "constexpr"},
    constexprs={"BLOCK": N},
    args=[Arg(a, IN, "float32", (N,)) for a in ("u", "y", "v", "p", "g", "x", "z")]
         + [Arg(a, OUT, "float32", (N,)) for a in ("as_", "ac", "ec", "ei", "nc", "lg", "tg",
                                                    "pw")],
    grid=(1,), reference=reference, make_inputs=make_inputs,
    rtol=1e-4, atol=1e-5,
)
