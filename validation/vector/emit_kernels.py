#!/usr/bin/env python3
"""shapes.csv and one kernel per measured v6e softmax-grid / load-series shape, blocked as XLA blocks it: a program
covers at most 2^20 elements, XLA's v6e fusion window (add1 3072x1024 compiles to output_window_bounds [128, 8] of
T(8,128) tiles = 1024x1024 elements, iteration_bounds [3, 1]); a C not a power of two is split in two, R=1 stays a row."""
import argparse
import csv
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SOFTMAX_CSV = os.path.join(HERE, "tpu_v6e_softmax_torchxla.csv")
LOAD_CSV = os.path.join(HERE, "tpu_v6e_load_torchxla.csv")
ARCHIVE_REPO = "https://github.com/PSAL-POSTECH/tpu-perf-archive"
LOAD_PROGRAMS = ("max", "sum", "add1", "slice")
BUDGET = 1 << 20
LANES = 128
SHAPE_COLS = ["table", "program", "R", "C", "kernel", "form", "BM", "BC"]

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


def p2(n):
    """The smallest power of two not below n."""
    return 1 << (n - 1).bit_length()


def rows_per_block(R, C, budget=BUDGET):
    """Rows of one block: the largest power of two dividing R whose block of C columns fits the budget."""
    cap = 1 << (max(1, budget // C).bit_length() - 1)
    b = 1
    while b * 2 <= cap and R % (b * 2) == 0:
        b *= 2
    return b


def _softmax(form, body, grid, BM, BC, doc):
    return dict(form=form, name="softmax_kernel", body=body, grid=grid, BM=BM, BC=BC, odt="bf16",
                otorch="bfloat16", oshape="R * C", ref=SOFTMAX_REF, doc=doc)


def softmax_rows(R, C):
    """Rows on the lanes, each row whole, C padded to the next power of two and masked."""
    BC = p2(C)
    BM = rows_per_block(R, BC)
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


def softmax_split(R, C):
    """Rows on the lanes, C = A + B in two power-of-two pieces read and written apart, nothing padded."""
    A = 1 << (C.bit_length() - 1)
    B = C - A
    BM = rows_per_block(R, C)
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


def softmax_chunks(R, C):
    """Fewer rows than lanes: one program, a [128, R, K] block whose lane axis is 128 chunks of C; K = 1 drops
    its axis, since a reduce over a unit axis before the lanes does not lower (reduce_unit_axis_then_lanes.py)."""
    K = 1
    while -(-C // K) > LANES:
        K *= 2
    if K == 1 and C == LANES:
        return softmax_lanes(R, C)
    L = C // K
    mo = "" if L == LANES else f", mask=q[:, None, None] < {L}, other=-1.0e30"
    m = "" if L == LANES else f", mask=q[:, None, None] < {L}"
    body = f'''    def softmax_kernel(x_ptr, o_ptr):
        q = tl.arange(0, {LANES})
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
                    f"softmax {R}x{C}: one program, C as {LANES} chunks of {K} on the lanes, {R} rows.")


def softmax_lanes(R, C):
    """Fewer rows than lanes and C = 128: one program, a [128, R] block with C on the lanes."""
    body = f'''    def softmax_kernel(x_ptr, o_ptr):
        q = tl.arange(0, {LANES})
        r = tl.arange(0, {R})
        idx = r[None, :] * {C} + q[:, None]
        x = tl.load(x_ptr + idx).to(tl.float32)
        mx = tl.max(x, axis=0)
        e = tl.exp(x - mx[None, :])
        s = tl.sum(e, axis=0)
        tl.store(o_ptr + idx, (e / s[None, :]).to(tl.bfloat16))
'''
    return _softmax("chunks", body, "1", R, 1, f"softmax {R}x{C}: one program, C on the {LANES} lanes, {R} rows.")


def whole(op):
    """max or sum of all of x: one program walks budget-sized row blocks; the scalar is stored as f32,
    since a scalar f32 -> bf16 store does not link (tests/cases/mechanism/ops/ops_scalar_bf16_store.py)."""
    def form(R, C):
        BC = p2(C)
        BM = rows_per_block(R, BC)
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


def add1(R, C):
    """x + 1, bf16 in and out, one budget-sized block of rows per program."""
    BC = p2(C)
    BM = rows_per_block(R, BC)
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


def slice1(R, C):
    """x[:1, :1] + 1: one element read, one f32 element written."""
    body = '''    def slice_kernel(x_ptr, o_ptr):
        tl.store(o_ptr, tl.load(x_ptr).to(tl.float32) + 1.0)
'''
    return dict(form="one", name="slice_kernel", body=body, grid="1", BM=1, BC=1, odt="fp32",
                otorch="float32", oshape="1", ref="(x[:1, :1].float() + 1).reshape(-1)",
                doc=f"x[:1, :1] + 1 of x {R}x{C} bf16 as f32: one element.")


def softmax(R, C):
    """Rule set A: a single row keeps the rows form, fewer rows than lanes spread C over the lanes,
    and from 128 rows on a C that is not a power of two is split instead of padded."""
    if R == 1:
        return softmax_rows(R, C)
    if R < LANES:
        return softmax_chunks(R, C)
    return softmax_rows(R, C) if C == p2(C) else softmax_split(R, C)


FORMS = {"softmax": softmax, "max": whole("max"), "sum": whole("sum"), "add1": add1, "slice": slice1}


def measured_shapes(softmax_csv=SOFTMAX_CSV, load_csv=LOAD_CSV):
    """(program, R, C) of every softmax grid row and every load series row of the copied tables."""
    with open(softmax_csv, newline="") as fh:
        out = [("softmax", int(r["R"]), int(r["C"])) for r in csv.DictReader(fh)]
    with open(load_csv, newline="") as fh:
        out += [(r["program"], int(r["R"]), int(r["C"])) for r in csv.DictReader(fh)]
    return out


def render(program, R, C):
    """The shapes.csv row and the kernel file text of one shape."""
    d = FORMS[program](R, C)
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


def main():
    """Command line: optionally refresh the measured rows from an archive checkout, then write
    shapes.csv and kernels/ for every measured shape."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", help="a tpu-perf-archive checkout to copy the measured rows from first")
    ap.add_argument("--out", default=HERE, help="shapes.csv and kernels/ go here")
    args = ap.parse_args()
    if args.archive:
        copy_archive(args.archive)
    os.makedirs(os.path.join(args.out, "kernels"), exist_ok=True)
    rows = []
    for program, R, C in measured_shapes():
        row, text = render(program, R, C)
        with open(os.path.join(args.out, "kernels", row["kernel"] + ".py"), "w") as fh:
            fh.write(text)
        rows.append(row)
    with open(os.path.join(args.out, "shapes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHAPE_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} kernels -> {os.path.join(args.out, 'kernels')}")


if __name__ == "__main__":
    main()
