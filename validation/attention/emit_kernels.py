#!/usr/bin/env python3
"""OUT/shapes.csv and one kernel per measured v6e attention shape in OUT/kernels: prefill is causal flash attention as
splash runs it, decode is RPA's paged decode; every block comes from the tiling policy (validation/tiling.py, or
--tiling). CI times a shape iff its score elements (cost) are at most CI_CAP = 2^26; shapes.csv says which (ci)."""
import argparse
import csv
import importlib.util
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ATTENTION_CSV = os.path.join(HERE, "tpu_v6e_attention_torchxla.csv")
ARCHIVE_REPO = "https://github.com/PSAL-POSTECH/tpu-perf-archive"
SHAPE_COLS = ["kind", "Hq", "Hkv", "S", "kernel", "form", "BQ", "BKV", "GP", "cost", "ci"]
TILING = os.path.join(os.path.dirname(HERE), "tiling.py")
D = 128
CI_CAP = 1 << 26


def use_tiling(path=TILING):
    """Load the tiling policy at `path` as this emitter's `tiling`; every block of every kernel comes from it."""
    global tiling
    spec = importlib.util.spec_from_file_location("validation_tiling", path)
    tiling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tiling)
    return tiling


use_tiling()

HEAD = '''"""{doc} (validation/attention/emit_kernels.py)"""
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

HQ, HKV, S, D = {Hq}, {Hkv}, {S}, {D}
G = HQ // HKV
SCALE = {scale!r}


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
{body}
    return attention_kernel


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {{x: torch.randn(n, generator=g, dtype=torch.float32).to(torch.bfloat16) for x, n in {inputs}}}


def reference(inputs):
    import torch
{ref}


SPEC = KernelSpec(
    name="attention_kernel",
    kernel=kernel,
    signature={signature},
    constexprs={{}},
    args=[{args}],
    grid={grid},
    reference=reference,
    make_inputs=make_inputs,
    rtol=2e-2, atol=2e-2,
)
'''

PREFILL_REF = '''    q = inputs["q"].float().reshape(HQ, S, D)
    k = inputs["k"].float().reshape(HKV, S, D).repeat_interleave(G, 0)
    v = inputs["v"].float().reshape(HKV, S, D).repeat_interleave(G, 0)
    s = (q @ k.transpose(1, 2) * SCALE).masked_fill(torch.ones(S, S, dtype=torch.bool).triu(1), float("-inf"))
    return {"o": (torch.softmax(s, -1) @ v).to(torch.bfloat16).reshape(-1)}'''

DECODE_REF = '''    q = inputs["q"].float().reshape(HQ, 1, D)
    kv = inputs["kv"].float().reshape(S, HKV, 2, D)
    k = kv[:, :, 0].transpose(0, 1).repeat_interleave(G, 0)
    v = kv[:, :, 1].transpose(0, 1).repeat_interleave(G, 0)
    return {"o": (torch.softmax(q @ k.transpose(1, 2) * SCALE, -1) @ v).to(torch.bfloat16).reshape(-1)}'''

STEP = '''{i}qk = tl.dot(q, kt, out_dtype=tl.float32) * {scale!r}
{mask}{i}m_new = tl.maximum(m_i, tl.max(qk, axis=1))
{i}alpha = tl.exp(m_i - m_new)
{i}p = tl.exp(qk - m_new[:, None])
{i}l_i = l_i * alpha + tl.sum(p, axis=1)
{i}acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
{i}m_i = m_new
'''


def _step(indent, mask=""):
    """One online-softmax update of (m_i, l_i, acc) by the KV block kt, v: two tl.dot, f32 state, p cast to bf16."""
    i = " " * indent
    return STEP.format(i=i, scale=1.0 / D ** 0.5, mask=f"{i}{mask}\n" if mask else "")


def prefill(Hq, Hkv, S, b):
    """splash at block BQ = BKV: per query head (GQA: KV head h // G), each query block runs the KV blocks left of the
    diagonal unmasked, the diagonal masked, none above; the query blocks are a static loop, not a grid axis: a TOG has
    no loop counted by the program id nor a DMA under an if (tests/cases/mechanism/loops/loop_08_*, loop_09_*)."""
    BQ, BKV = b["BQ"], b["BKV"]
    body = f'''    def attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr):
        h = tl.program_id(0)
        kbase = (h // {Hq // Hkv}) * {S * D}
        rn = tl.arange(0, {BKV})
        rd = tl.arange(0, {D})
        for qb in tl.static_range({S // BQ}):
            rm = qb * {BQ} + tl.arange(0, {BQ})
            q = tl.load(q_ptr + h * {S * D} + rm[:, None] * {D} + rd[None, :])
            m_i = tl.full(({BQ},), -1.0e30, dtype=tl.float32)
            l_i = tl.zeros(({BQ},), dtype=tl.float32)
            acc = tl.zeros(({BQ}, {D}), dtype=tl.float32)
            for start in range(0, qb * {BQ}, {BKV}):
                n = start + rn
                kt = tl.load(k_ptr + kbase + n[None, :] * {D} + rd[:, None])
                v = tl.load(v_ptr + kbase + n[:, None] * {D} + rd[None, :])
{_step(16)}            n = qb * {BQ} + rn
            kt = tl.load(k_ptr + kbase + n[None, :] * {D} + rd[:, None])
            v = tl.load(v_ptr + kbase + n[:, None] * {D} + rd[None, :])
{_step(12, "qk = tl.where(rm[:, None] >= n[None, :], qk, -1.0e30)")}            tl.store(o_ptr + h * {S * D} + rm[:, None] * {D} + rd[None, :], (acc / l_i[:, None]).to(tl.bfloat16))
'''
    return dict(body=body, grid=f"({Hq},)", ref=PREFILL_REF,
                inputs='(("q", HQ * S * D), ("k", HKV * S * D), ("v", HKV * S * D))',
                signature='{"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "o_ptr": "*bf16"}',
                args='Arg("q", IN, "bfloat16", (HQ * S * D,)), Arg("k", IN, "bfloat16", (HKV * S * D,)), '
                     'Arg("v", IN, "bfloat16", (HKV * S * D,)), Arg("o", OUT, "bfloat16", (HQ * S * D,))',
                doc=f"causal prefill {Hq}/{Hkv} heads, S {S}, head_dim {D}, bf16: splash at block {BQ}, "
                    f"the {S // BQ} query block(s) a static loop in each of {Hq} head programs.")


def decode(Hq, Hkv, S, b):
    """RPA decode: one query token, K and V interleaved per KV head in pages of 128 tokens ([S/128, 128, 2*Hkv, 128]),
    one program per KV head holding its G query heads as GP rows (masked past G), BKV tokens a step; the pages are
    contiguous (the measurement's page table is the identity), so a block of pages is one strided load."""
    GP, BKV, CH = b["GP"], b["BKV"], 2 * Hkv * D
    G = Hq // Hkv
    qm = "" if GP == G else f", mask=rg[:, None] < {G}"
    body = f'''    def attention_kernel(q_ptr, kv_ptr, o_ptr):
        kh = tl.program_id(0)
        rg = tl.arange(0, {GP})
        rd = tl.arange(0, {D})
        q = tl.load(q_ptr + (kh * {G} + rg[:, None]) * {D} + rd[None, :]{qm}{", other=0.0" if qm else ""})
        m_i = tl.full(({GP},), -1.0e30, dtype=tl.float32)
        l_i = tl.zeros(({GP},), dtype=tl.float32)
        acc = tl.zeros(({GP}, {D}), dtype=tl.float32)
        for start in range(0, {S}, {BKV}):
            t = start + tl.arange(0, {BKV})
            kt = tl.load(kv_ptr + t[None, :] * {CH} + (2 * kh) * {D} + rd[:, None])
            v = tl.load(kv_ptr + t[:, None] * {CH} + (2 * kh + 1) * {D} + rd[None, :])
{_step(12)}        tl.store(o_ptr + (kh * {G} + rg[:, None]) * {D} + rd[None, :], (acc / l_i[:, None]).to(tl.bfloat16){qm})
'''
    return dict(body=body, grid=f"({Hkv},)", ref=DECODE_REF,
                inputs='(("q", HQ * D), ("kv", S * 2 * HKV * D))',
                signature='{"q_ptr": "*bf16", "kv_ptr": "*bf16", "o_ptr": "*bf16"}',
                args='Arg("q", IN, "bfloat16", (HQ * D,)), Arg("kv", IN, "bfloat16", (S * 2 * HKV * D,)), '
                     'Arg("o", OUT, "bfloat16", (HQ * D,))',
                doc=f"decode {Hq}/{Hkv} heads, KV {S} tokens in pages of 128, head_dim {D}, bf16: RPA, "
                    f"{BKV} tokens a step, {G} query heads of a KV head as {GP} rows.")


FORMS = {"prefill": prefill, "decode": decode}


def cost(kind, Hq, Hkv, S, b):
    """Score elements the kernel computes, the number the pool runs largest first and CI_CAP bounds: prefill the
    causal block pairs times BQ*BKV per query head, decode GP*S per KV head."""
    if kind == "prefill":
        nb = S // b["BQ"]
        return Hq * nb * (nb + 1) // 2 * b["BQ"] * b["BKV"]
    return Hkv * b["GP"] * S


def measured_shapes(attention_csv=ATTENTION_CSV):
    """(kind, Hq, Hkv, S) of every row of the copied table."""
    with open(attention_csv, newline="") as fh:
        return [(r["kind"], int(r["Hq"]), int(r["Hkv"]), int(r["S"])) for r in csv.DictReader(fh)]


def render(kind, Hq, Hkv, S):
    """The shapes.csv row and the kernel file text of one shape."""
    b = tiling.attention_blocks(kind, Hq, Hkv, S)
    d = FORMS[kind](Hq, Hkv, S, b)
    c = cost(kind, Hq, Hkv, S, b)
    row = {"kind": kind, "Hq": Hq, "Hkv": Hkv, "S": S, "kernel": f"{kind}_{Hq}x{Hkv}_s{S}", "form": b["form"],
           "BQ": b["BQ"], "BKV": b["BKV"], "GP": b["GP"], "cost": c, "ci": int(c <= CI_CAP)}
    return row, HEAD.format(Hq=Hq, Hkv=Hkv, S=S, D=D, scale=1.0 / D ** 0.5, **d)


def copy_archive(archive):
    """Copy the attention rows out of a tpu-perf-archive checkout, naming its commit."""
    commit = subprocess.check_output(["git", "-C", archive, "rev-parse", "--short=7", "HEAD"], text=True).strip()
    src = os.path.join(archive, "data/v6e/attention/torchxla.csv")
    with open(src, newline="") as fh:
        rd = csv.DictReader(fh)
        rows = [dict(r, archive_repo=ARCHIVE_REPO, archive_commit=commit) for r in rd]
        cols = rd.fieldnames + ["archive_repo", "archive_commit"]
    with open(ATTENTION_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows of {os.path.relpath(src, archive)} at {commit} -> {os.path.basename(ATTENTION_CSV)}")


def generate(out):
    """Write out/kernels and out/shapes.csv for every measured shape."""
    os.makedirs(os.path.join(out, "kernels"), exist_ok=True)
    rows = []
    for shape in measured_shapes():
        row, text = render(*shape)
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
    print(f"{len(rows)} kernels -> {os.path.join(args.out, 'kernels')}, {sum(r['ci'] for r in rows)} timed in CI")


if __name__ == "__main__":
    main()
