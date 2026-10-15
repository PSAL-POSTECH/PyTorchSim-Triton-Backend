#!/usr/bin/env python3
"""One kernel per shapes.csv row (M, K, N, form) into OUT/kernels, and OUT/shapes.csv with the kernel each row got:
every block comes from the tiling policy (validation/tiling.py, or --tiling), this file only holds the form
templates, so a tiling change edits the policy alone."""
import argparse
import csv
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = os.path.join(HERE, "shapes.csv")
TILING = os.path.join(os.path.dirname(HERE), "tiling.py")


def use_tiling(path=TILING):
    """Load the tiling policy at `path` as this emitter's `tiling`; every block of every kernel comes from it."""
    global tiling
    spec = importlib.util.spec_from_file_location("validation_tiling", path)
    tiling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tiling)
    return tiling


use_tiling()

HEAD = '''"""{doc}"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = {M}, {N}, {K}
{consts}

def kernel():
    import triton
    import triton.language as tl

    @triton.jit
{body}
    return gemm_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {{
        "a": torch.randn(M, K, generator=g, dtype=torch.float32).to(torch.bfloat16),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32).to(torch.bfloat16),
    }}


def reference(inputs):
    import torch
    return {{"c": torch.matmul(inputs["a"].to(torch.float32), inputs["b"].to(torch.float32)).to(torch.bfloat16)}}


SPEC = KernelSpec(
    name="gemm_kernel",
    kernel=kernel,
    signature={signature},
    constexprs={constexprs},
    args=[
        Arg("a", IN, "bfloat16", (M, K)),
        Arg("b", IN, "bfloat16", (K, N)),
        Arg("c", OUT, "bfloat16", (M, N)),
    ],
    grid={grid},
    reference=reference,
    make_inputs=make_inputs,
    rtol={tol}, atol={tol},
)
'''
ONE = dict(consts="", signature='{"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16"}', constexprs="{}",
           grid="(1, 1)", tol="2e-2")


def _ranges(sizes, indent=8):
    return "".join(f"{' ' * indent}r{s} = tl.arange(0, {s})\n" for s in sorted(set(sizes)))


def _offsets(pieces):
    return [(sum(pieces[:i]), p) for i, p in enumerate(pieces)]


def _kernel(name, doc, body, M, K, N, **kw):
    return name, HEAD.format(doc=doc, M=M, N=N, K=K, body=body, **dict(ONE, **kw))


def gemv(M, K, N, b):
    """M = 1 as PyTorchSim emits it (rewrite_row_mm, Inductor reduction): N in XBLOCK-wide programs, K summed into an
    [XBLOCK, R0_BLOCK] accumulator in R0_BLOCK steps and reduced after the loop; r0 masked when R0_BLOCK does not divide K."""
    X, R = b["XBLOCK"], b["R0_BLOCK"]
    mask = K % R != 0
    lm, lo = (", r0_mask", ", other=0.0") if mask else ("", "")
    body = ("    def gemm_kernel(a_ptr, b_ptr, c_ptr):\n"
            f"        xindex = tl.program_id(0) * {X} + tl.arange(0, {X})[:, None]\n"
            f"        r0_base = tl.arange(0, {R})[None, :]\n"
            f"        _acc = tl.full([{X}, {R}], 0, tl.float32)\n"
            f"        for r0_offset in range(0, {K}, {R}):\n"
            "            r0_index = r0_offset + r0_base\n"
            + (f"            r0_mask = r0_index < {K}\n" if mask else "")
            + f"            x = tl.load(a_ptr + r0_index{lm}{lo}).to(tl.float32)\n"
            f"            w = tl.load(b_ptr + (xindex + {N} * r0_index){lm}{lo}).to(tl.float32)\n"
            + (f"            _acc = tl.where(r0_mask, _acc + x * w, _acc)\n" if mask else "            _acc = _acc + x * w\n")
            + "        tl.store(c_ptr + xindex, tl.sum(_acc, 1)[:, None].to(tl.bfloat16))\n")
    return _kernel(f"pts_{M}x{K}x{N}_x{X}r{R}",
                   f"gemm {M}x{K}x{N} (MxKxN) bf16 as PyTorchSim {tiling.PYTORCHSIM} emits it: XBLOCK={X}, R0_BLOCK={R}.",
                   body, M, K, N, grid=f"({N // X}, 1)")


def res_b(M, K, N, b):
    """B resident: B read once as N pieces of the padded K, then W-row blocks of A, one tl.dot per piece."""
    Kp, W, pieces = b["Kp"], b["W"], b["n_pieces"]
    mb = "" if Kp == K else f", mask=rk[:, None] < {K}, other=0.0"
    ma = "" if Kp == K else f", mask=rk[None, :] < {K}, other=0.0"
    body = f"    def gemm_kernel(a_ptr, b_ptr, c_ptr):\n        rk = tl.arange(0, {Kp})\n" + _ranges([W] + pieces)
    body += "".join(f"        b_{j} = tl.load(b_ptr + rk[:, None] * {N} + ({o} + r{n})[None, :]{mb})\n"
                    for j, (o, n) in enumerate(_offsets(pieces)))
    body += (f"        for m in range(0, {M}, {W}):\n            om = m + r{W}\n"
             f"            a = tl.load(a_ptr + om[:, None] * {K} + rk[None, :]{ma})\n")
    body += "".join(f"            acc = tl.dot(a, b_{j}, out_dtype=tl.float32)\n"
                    f"            tl.store(c_ptr + om[:, None] * {N} + ({o} + r{n})[None, :], acc.to(tl.bfloat16))\n"
                    for j, (o, n) in enumerate(_offsets(pieces)))
    return _kernel(f"tk_{M}x{K}x{N}_b_w{W}c{min(pieces)}",
                   f"gemm {M}x{K}x{N} (MxKxN) bf16, mode=b Kwhole Npieces={'+'.join(map(str, pieces))} window={W}.",
                   body, M, K, N)


def res_a(M, K, N, b):
    """A resident: A read once as Mc-row pieces of the padded K, then Wn-column windows of B."""
    Kp, Wn, Mc = b["Kp"], b["Wn"], b["Mc"]
    ma = "" if Kp == K else f", mask=ok[None, :] < {K}, other=0.0"
    mb = "" if Kp == K else f", mask=ok[:, None] < {K}, other=0.0"
    rows = range(0, M, Mc)
    body = (f"    def gemm_kernel(a_ptr, b_ptr, c_ptr):\n        ok = tl.arange(0, {Kp})\n"
            f"        om = tl.arange(0, {Mc})\n")
    body += "".join(f"        a_{i} = tl.load(a_ptr + ({m} + om)[:, None] * {K} + ok[None, :]{ma})\n"
                    for i, m in enumerate(rows))
    body += (f"        for n in range(0, {N}, {Wn}):\n            on = n + tl.arange(0, {Wn})\n"
             f"            b = tl.load(b_ptr + ok[:, None] * {N} + on[None, :]{mb})\n")
    body += "".join(f"            acc = tl.dot(a_{i}, b, out_dtype=tl.float32)\n"
                    f"            tl.store(c_ptr + ({m} + om)[:, None] * {N} + on[None, :], acc.to(tl.bfloat16))\n"
                    for i, m in enumerate(rows))
    return _kernel(f"tk_{M}x{K}x{N}_a_w{Wn}c{Mc}",
                   f"gemm {M}x{K}x{N} (MxKxN) bf16, mode=a Kwhole Npieces=Wn{Wn} window={Wn}.", body, M, K, N)


def res_a_kpieces(M, K, N, b):
    """A resident with K in pieces (K whole overflows): each Mc-row piece of A is held as its K pieces and each
    Wn window of B likewise, the tl.dot of every K piece summed."""
    kp, Wn, Mc = _offsets(b["k_pieces"]), b["Wn"], b["Mc"]
    rows = range(0, M, Mc)
    body = "    def gemm_kernel(a_ptr, b_ptr, c_ptr):\n" + _ranges(b["k_pieces"]) + f"        om = tl.arange(0, {Mc})\n"
    body += "".join(f"        a_{i}_{p} = tl.load(a_ptr + ({m} + om)[:, None] * {K} + ({ko} + r{ks})[None, :])\n"
                    for i, m in enumerate(rows) for p, (ko, ks) in enumerate(kp))
    body += f"        for n in range(0, {N}, {Wn}):\n            on = n + tl.arange(0, {Wn})\n"
    body += "".join(f"            b_{p} = tl.load(b_ptr + ({ko} + r{ks})[:, None] * {N} + on[None, :])\n"
                    for p, (ko, ks) in enumerate(kp))
    for i, m in enumerate(rows):
        dots = " + ".join(f"tl.dot(a_{i}_{p}, b_{p}, out_dtype=tl.float32)" for p in range(len(kp)))
        body += (f"            acc = {dots}\n"
                 f"            tl.store(c_ptr + ({m} + om)[:, None] * {N} + on[None, :], acc.to(tl.bfloat16))\n")
    pieces = "+".join(str(s) for _, s in kp)
    return _kernel(f"tk_{M}x{K}x{N}_ak_w{Wn}c{Mc}",
                   f"gemm {M}x{K}x{N} (MxKxN) bf16, mode=ak Kwhole Npieces=Wn{Wn}Kpieces{pieces} window={Wn}.",
                   body, M, K, N)


GRID_BODY = '''    def gemm_kernel(a_ptr, b_ptr, c_ptr,
                    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)

        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))
'''


def tiled(M, K, N, b):
    """Inputs too big for the scratchpad: BM x BN output tiles over BK steps of K, walked in one program
    (template loop) or one program per tile with the sizes as constexprs (template grid)."""
    BM, BN, BK = b["BM"], b["BN"], b["BK"]
    if b["template"] == "grid":
        return _kernel(f"gemm_bf16out_{M}x{N}x{K}_b{BM}x{BN}x{BK}", f"gemm {M}x{N}x{K} bf16, block {BM}x{BN}x{BK}.",
                       GRID_BODY, M, K, N, consts=f"BM, BN, BK = {BM}, {BN}, {BK}\n",
                       signature='{"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16",\n'
                                 '               "M": "constexpr", "N": "constexpr", "K": "constexpr",\n'
                                 '               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"}',
                       constexprs='{"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK}',
                       grid="(M // BM, N // BN)", tol="1e-2")
    body = (f"    def gemm_kernel(a_ptr, b_ptr, c_ptr):\n        om = tl.arange(0, {BM})\n"
            f"        on = tl.arange(0, {BN})\n        ok = tl.arange(0, {BK})\n"
            f"        for m in range(0, {M}, {BM}):\n            for n in range(0, {N}, {BN}):\n"
            f"                acc = tl.zeros(({BM}, {BN}), dtype=tl.float32)\n"
            f"                for k in range(0, {K}, {BK}):\n"
            f"                    a = tl.load(a_ptr + (m + om)[:, None] * {K} + (k + ok)[None, :])\n"
            f"                    b = tl.load(b_ptr + (k + ok)[:, None] * {N} + (n + on)[None, :])\n"
            f"                    acc += tl.dot(a, b, out_dtype=tl.float32)\n"
            f"                tl.store(c_ptr + (m + om)[:, None] * {N} + (n + on)[None, :], acc.to(tl.bfloat16))\n")
    return _kernel(f"gx_{M}x{K}x{N}_tiled_w{BM}x{BN}x{BK}", f"gemm {M}x{K}x{N} (MxKxN) bf16, mode=tiled window={BM}.",
                   body, M, K, N)


def form_of(form):
    """The template of a shapes.csv form (the K-pieces form names its pieces, which the policy derives)."""
    return res_a_kpieces if form.startswith("res_a(Kpieces") else FORMS[form]


FORMS = {"gemv": gemv, "res_b(Kwhole)": res_b, "res_a(Kwhole)": res_a, "tiled_fallback": tiled}


def render(row):
    """The kernel name and file text of one shapes.csv row."""
    M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
    return form_of(row["form"])(M, K, N, tiling.gemm_blocks(M, K, N, row["form"]))


def generate(out, shapes=SHAPES):
    """Write every row's kernel under out/kernels and out/shapes.csv with the kernel column as emitted."""
    with open(shapes, newline="") as fh:
        rd = csv.DictReader(fh)
        rows, cols = list(rd), list(rd.fieldnames)
    if "kernel" not in cols:
        cols.insert(cols.index("N") + 1, "kernel")
    os.makedirs(os.path.join(out, "kernels"), exist_ok=True)
    for r in rows:
        r["kernel"], text = render(r)
        with open(os.path.join(out, "kernels", r["kernel"] + ".py"), "w") as fh:
            fh.write(text)
    with open(os.path.join(out, "shapes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    """Command line: the output directory, the shape list and the tiling policy."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--out", required=True, help="shapes.csv and kernels/ go here")
    ap.add_argument("--tiling", default=TILING, help="the tiling policy file (default validation/tiling.py)")
    args = ap.parse_args()
    use_tiling(args.tiling)
    rows = generate(args.out, args.shapes)
    print(f"{len(rows)} kernels -> {os.path.join(args.out, 'kernels')}")


if __name__ == "__main__":
    main()
