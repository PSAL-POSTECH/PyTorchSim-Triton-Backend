#!/usr/bin/env python3
"""OUT/shapes.csv and one kernel per measured v6e softmax-grid / load-series shape in OUT/kernels: the form and every
block come from the tiling policy (validation/tiling.py, or --tiling), this file only holds the form templates, so a
tiling change edits the policy alone."""
import argparse
import csv
import importlib.util
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SOFTMAX_CSV = os.path.join(HERE, "tpu_v6e_softmax_torchxla.csv")
LOAD_CSV = os.path.join(HERE, "tpu_v6e_load_torchxla.csv")
ARCHIVE_REPO = "https://github.com/PSAL-POSTECH/tpu-perf-archive"
LOAD_PROGRAMS = ("max", "sum", "add1", "slice")
SHAPE_COLS = ["table", "program", "R", "C", "kernel", "form", "BM", "BC"]
TILING = os.path.join(os.path.dirname(HERE), "tiling.py")


def use_tiling(path=TILING):
    """Load the tiling policy at `path` as this emitter's `tiling`; every block of every kernel comes from it."""
    global tiling
    spec = importlib.util.spec_from_file_location("validation_tiling", path)
    tiling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tiling)
    return tiling


use_tiling()

HEAD = '''"""{doc} Blocked as XLA blocks it (validation/vector/emit_kernels.py)."""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

R, C, BM, BC = {R}, {C}, {BM}, {BC}


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
{body}
    return {name}


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {{"x": torch.randn(R * C, generator=g, dtype=torch.float32).to(torch.bfloat16)}}


def reference(inputs):
    import torch
    x = inputs["x"].reshape(R, C)
    return {{"o": {ref}}}


SPEC = KernelSpec(
    name="{name}",
    kernel=kernel,
    signature={{"x_ptr": "*bf16", "o_ptr": "*{odt}"}},
    constexprs={{}},
    args=[Arg("x", IN, "bfloat16", (R * C,)), Arg("o", OUT, "{otorch}", ({oshape},))],
    grid=({grid},),
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
)
'''
SOFTMAX_REF = "torch.softmax(x, dim=-1, dtype=torch.float32).to(torch.bfloat16).reshape(-1)"


def _softmax(form, body, grid, BM, BC, doc):
    return dict(form=form, name="softmax_kernel", body=body, grid=grid, BM=BM, BC=BC, odt="bf16",
                otorch="bfloat16", oshape="R * C", ref=SOFTMAX_REF, doc=doc)


def softmax_rows(R, C, b):
    """Rows on the lanes, each row whole, C padded to BC and masked."""
    BM, BC = b["BM"], b["BC"]
    m = "" if C == BC else f", mask=cols[None, :] < {C}, other=-1.0e30"
    s = "" if C == BC else f", mask=cols[None, :] < {C}"
    body = f'''    def softmax_kernel(x_ptr, o_ptr):
        rows = tl.program_id(0) * {BM} + tl.arange(0, {BM})
        cols = tl.arange(0, {BC})
        idx = rows[:, None] * {C} + cols[None, :]
        x = tl.load(x_ptr + idx{m}).to(tl.float32)
        mx = tl.max(x, axis=1)
        e = tl.exp(x - mx[:, None])
        s = tl.sum(e, axis=1)
        tl.store(o_ptr + idx, (e / s[:, None]).to(tl.bfloat16){s})
'''
    return _softmax("rows", body, f"{R // BM}", BM, BC,
                    f"softmax {R}x{C}: {BM} rows per program, C padded to {BC}.")


def softmax_split(R, C, b):
    """Rows on the lanes, C = A + B in two pieces read and written apart, nothing padded."""
    BM, A, B = b["BM"], b["A"], b["B"]
    body = f'''    def softmax_kernel(x_ptr, o_ptr):
        rows = tl.program_id(0) * {BM} + tl.arange(0, {BM})
        ia = rows[:, None] * {C} + tl.arange(0, {A})[None, :]
        ib = rows[:, None] * {C} + {A} + tl.arange(0, {B})[None, :]
        xa = tl.load(x_ptr + ia).to(tl.float32)
        xb = tl.load(x_ptr + ib).to(tl.float32)
        mx = tl.maximum(tl.max(xa, axis=1), tl.max(xb, axis=1))
        ea = tl.exp(xa - mx[:, None])
        eb = tl.exp(xb - mx[:, None])
        s = tl.sum(ea, axis=1) + tl.sum(eb, axis=1)
        tl.store(o_ptr + ia, (ea / s[:, None]).to(tl.bfloat16))
        tl.store(o_ptr + ib, (eb / s[:, None]).to(tl.bfloat16))
'''
    return _softmax("split", body, f"{R // BM}", BM, A,
                    f"softmax {R}x{C}: {BM} rows per program, C = {A} + {B}.")


def softmax_chunks(R, C, b):
    """One program, a [Q, R, K] block whose lane axis is Q chunks of C, L of them live; the lanes form drops K,
    since a reduce over a unit axis before the lanes does not lower (reduce_unit_axis_then_lanes.py)."""
    Q, K, L = b["Q"], b["K"], b["L"]
    mo = "" if L == Q else f", mask=q[:, None, None] < {L}, other=-1.0e30"
    m = "" if L == Q else f", mask=q[:, None, None] < {L}"
    body = f'''    def softmax_kernel(x_ptr, o_ptr):
        q = tl.arange(0, {Q})
        r = tl.arange(0, {R})
        k = tl.arange(0, {K})
        idx = r[None, :, None] * {C} + q[:, None, None] * {K} + k[None, None, :]
        x = tl.load(x_ptr + idx{mo}).to(tl.float32)
        mx = tl.max(tl.max(x, axis=2), axis=0)
        e = tl.exp(x - mx[None, :, None])
        s = tl.sum(tl.sum(e, axis=2), axis=0)
        tl.store(o_ptr + idx, (e / s[None, :, None]).to(tl.bfloat16){m})
'''
    return _softmax("chunks", body, "1", R, K,
                    f"softmax {R}x{C}: one program, C as {Q} chunks of {K} on the lanes, {R} rows.")


def softmax_lanes(R, C, b):
    """C = Q: one program, a [Q, R] block with C on the lanes."""
    Q = b["Q"]
    body = f'''    def softmax_kernel(x_ptr, o_ptr):
        q = tl.arange(0, {Q})
        r = tl.arange(0, {R})
        idx = r[None, :] * {C} + q[:, None]
        x = tl.load(x_ptr + idx).to(tl.float32)
        mx = tl.max(x, axis=0)
        e = tl.exp(x - mx[None, :])
        s = tl.sum(e, axis=0)
        tl.store(o_ptr + idx, (e / s[None, :]).to(tl.bfloat16))
'''
    return _softmax("chunks", body, "1", R, 1, f"softmax {R}x{C}: one program, C on the {Q} lanes, {R} rows.")


def whole(op):
    """max or sum of all of x: one program walks budget-sized row blocks; the scalar is stored as f32,
    since a scalar f32 -> bf16 store does not link (tests/cases/mechanism/ops/ops_scalar_bf16_store.py)."""
    def form(R, C, b):
        BM, BC = b["BM"], b["BC"]
        init = "-1.0e30" if op == "max" else "0.0"
        m = "" if C == BC else f", mask=cols[None, :] < {C}, other={init}"
        step = "acc = tl.maximum(acc, x)" if op == "max" else "acc = acc + x"
        red = f"tl.{op}(tl.{op}(acc, axis=1), axis=0)"
        body = f'''    def {op}_kernel(x_ptr, o_ptr):
        rows = tl.arange(0, {BM})
        cols = tl.arange(0, {BC})
        acc = tl.full([{BM}, {BC}], {init}, tl.float32)
        for r in range(0, {R}, {BM}):
            x = tl.load(x_ptr + (r + rows)[:, None] * {C} + cols[None, :]{m}).to(tl.float32)
            {step}
        tl.store(o_ptr, {red})
'''
        ref = "torch.max(x).reshape(1).float()" if op == "max" else "x.float().sum().reshape(1)"
        return dict(form="walk", name=f"{op}_kernel", body=body, grid="1", BM=BM, BC=BC, odt="fp32",
                    otorch="float32", oshape="1", ref=ref,
                    doc=f"{op} of all of x {R}x{C} bf16 as f32: one program walking {BM}-row blocks.")
    return form


def add1(R, C, b):
    """x + 1, bf16 in and out, one block of BM rows per program."""
    BM, BC = b["BM"], b["BC"]
    m = "" if C == BC else f", mask=cols[None, :] < {C}"
    body = f'''    def add1_kernel(x_ptr, o_ptr):
        rows = tl.program_id(0) * {BM} + tl.arange(0, {BM})
        cols = tl.arange(0, {BC})
        idx = rows[:, None] * {C} + cols[None, :]
        x = tl.load(x_ptr + idx{m})
        tl.store(o_ptr + idx, x + 1.0{m})
'''
    return dict(form="rows", name="add1_kernel", body=body, grid=f"{R // BM}", BM=BM, BC=BC, odt="bf16",
                otorch="bfloat16", oshape="R * C", ref="(x + 1).reshape(-1)",
                doc=f"x + 1 over {R}x{C} bf16: {BM} rows per program.")


def slice1(R, C, b):
    """x[:1, :1] + 1: one element read, one f32 element written."""
    body = '''    def slice_kernel(x_ptr, o_ptr):
        tl.store(o_ptr, tl.load(x_ptr).to(tl.float32) + 1.0)
'''
    return dict(form="one", name="slice_kernel", body=body, grid="1", BM=1, BC=1, odt="fp32",
                otorch="float32", oshape="1", ref="(x[:1, :1].float() + 1).reshape(-1)",
                doc=f"x[:1, :1] + 1 of x {R}x{C} bf16 as f32: one element.")


FORMS = {("softmax", "rows"): softmax_rows, ("softmax", "split"): softmax_split, ("softmax", "chunks"): softmax_chunks,
         ("softmax", "lanes"): softmax_lanes, ("max", "walk"): whole("max"), ("sum", "walk"): whole("sum"),
         ("add1", "rows"): add1, ("slice", "one"): slice1}


def measured_shapes(softmax_csv=SOFTMAX_CSV, load_csv=LOAD_CSV):
    """(program, R, C) of every softmax grid row and every load series row of the copied tables."""
    with open(softmax_csv, newline="") as fh:
        out = [("softmax", int(r["R"]), int(r["C"])) for r in csv.DictReader(fh)]
    with open(load_csv, newline="") as fh:
        out += [(r["program"], int(r["R"]), int(r["C"])) for r in csv.DictReader(fh)]
    return out


def render(program, R, C):
    """The shapes.csv row and the kernel file text of one shape."""
    b = tiling.vector_blocks(program, R, C)
    d = FORMS[program, b["form"]](R, C, b)
    name = f"{program}_{R}x{C}"
    row = {"table": "softmax" if program == "softmax" else "load", "program": program, "R": R, "C": C,
           "kernel": name, "form": d["form"], "BM": d["BM"], "BC": d["BC"]}
    return row, HEAD.format(R=R, C=C, **d)


def copy_archive(archive):
    """Copy the rows this set compares against out of a tpu-perf-archive checkout, naming its commit."""
    commit = subprocess.check_output(["git", "-C", archive, "rev-parse", "--short=7", "HEAD"], text=True).strip()
    picks = ((os.path.join(archive, "data/v6e/softmax/torchxla.csv"), SOFTMAX_CSV,
              lambda r: "grid" in r["source"].split(";")),
             (os.path.join(archive, "data/v6e/load/torchxla.csv"), LOAD_CSV,
              lambda r: r["source"] == "series" and r["program"] in LOAD_PROGRAMS))
    for src, dst, keep in picks:
        with open(src, newline="") as fh:
            rd = csv.DictReader(fh)
            rows = [dict(r, archive_repo=ARCHIVE_REPO, archive_commit=commit) for r in rd if keep(r)]
            cols = rd.fieldnames + ["archive_repo", "archive_commit"]
        with open(dst, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows of {os.path.relpath(src, archive)} at {commit} -> {os.path.basename(dst)}")


def generate(out):
    """Write out/kernels and out/shapes.csv for every measured shape."""
    os.makedirs(os.path.join(out, "kernels"), exist_ok=True)
    rows = []
    for program, R, C in measured_shapes():
        row, text = render(program, R, C)
        with open(os.path.join(out, "kernels", row["kernel"] + ".py"), "w") as fh:
            fh.write(text)
        rows.append(row)
    with open(os.path.join(out, "shapes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHAPE_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    """Command line: optionally refresh the measured rows from an archive checkout, then write shapes.csv and
    kernels/ for every measured shape under the tiling policy."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", help="a tpu-perf-archive checkout to copy the measured rows from first")
    ap.add_argument("--out", required=True, help="shapes.csv and kernels/ go here")
    ap.add_argument("--tiling", default=TILING, help="the tiling policy file (default validation/tiling.py)")
    args = ap.parse_args()
    if args.archive:
        copy_archive(args.archive)
    use_tiling(args.tiling)
    rows = generate(args.out)
    print(f"{len(rows)} kernels -> {os.path.join(args.out, 'kernels')}")


if __name__ == "__main__":
    main()
