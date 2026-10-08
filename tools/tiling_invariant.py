"""Which sizes the compiler handles, and which the KERNEL SOURCE has to get right.

THE WORKING ASSUMPTION
----------------------
A kernel here is one core's worth of work, and its tile is assumed to FIT IN THE
SCRATCHPAD. The Triton source picks that size (BM/BN/BK, the grid, the tl.arange
extents) and we take it as given. What the compiler owns is everything below
that: turning one core's tile into units the hardware can actually issue.

So the question this tool answers is NOT "does the compiler tile everything" --
it does not, by design. It is "given a tile the source chose, which sizes go
through, and does anything wrong slip past".

WHAT THE COMPILER DOES SPLIT

  the lane axis     vectorize._lane_split -- the axis fit_layout_to_lanes
                    chose, across VECTORLANE_SIZE lanes
  the vector axis   legalize -- one instruction covers
                    VLEN_BITS * ISA_MAX_LMUL bits, divided by the widest
                    element the loop actually holds
  the systolic tile legalize.systolic_tile -- the array eats SS x SS x SS

An axis over the instruction width is cut BY NARROWING THE LOOP STEP: an
`affine.for 0 to 128 step 128` over vector<128> becomes `step 32` over
vector<32>, so the same bound now takes four trips. The bound never moves, which
is why this is a retype and not a reshape -- an earlier version expanded
[128,128] to [128,4,32] and split one linalg.reduce into two.

Splitting at all is a CORRECTNESS requirement, not a speed choice -- vsetvli
clamps vl silently (processor.cc:632), so an unsplit 128-wide reduction would
sum 32 elements and report no error.

WHAT IT DOES NOT SPLIT

  Capacity. A tile larger than the scratchpad is not streamed through in
  pieces; the DMA still moves the whole tile in one transfer. README Phase 5.
  The assumption is checked rather than merely assumed: spad.check_budget reads
  the .spad section after linking and raises SpadOverflow past SPAD_SIZE.

WHAT EACH PROBE MEANS
---------------------
  PASS + no error            -> this size goes through
  FAIL with a divisibility   -> the source has to pick a size that suits the
    or shape complaint          hardware. EXPECTED for anything the three
                                limits above cannot express -- what matters is
                                that it is refused BY NAME rather than mangled.
  PASS but wrong numbers     -> the only real failure. A size that compiles and
                                runs and is silently wrong means some limit is
                                not being checked at all.

The third line is the one to watch. The first two are a map of where the
source/compiler boundary currently sits; the third is a bug.

Run:  python tools/tiling_invariant.py            # shape probes, stage 3 only
      python tools/tiling_invariant.py --numeric  # also run Spike and compare
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The compiler driver. It was `run.py` at the repo root, which became
#: tests/run.py and then this; nothing runs tools/, so the old name stayed.
DRIVER = "pytorchsim-triton-compiler"
sys.path.insert(0, ROOT)

from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

LANES = machine.VECTORLANE_SIZE

TEMPLATE = '''\
"""Generated probe -- see tools/tiling_invariant.py."""
import sys
sys.path.insert(0, {root!r})
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

N = {n}
BLOCK = {block}


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def {name}(x_ptr, y_ptr, o_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(o_ptr + offs, x + y)

    return {name}


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {{
        "x": torch.randn(N, generator=g, dtype=torch.float32),
        "y": torch.randn(N, generator=g, dtype=torch.float32),
    }}


def reference(inputs):
    return {{"o": inputs["x"] + inputs["y"]}}


SPEC = KernelSpec(
    name={name!r},
    kernel=kernel,
    signature={{"x_ptr": "*fp32", "y_ptr": "*fp32", "o_ptr": "*fp32",
               "n": "constexpr", "BLOCK": "constexpr"}},
    constexprs={{"n": N, "BLOCK": BLOCK}},
    args=[
        Arg("x", IN, "float32", (N,)),
        Arg("y", IN, "float32", (N,)),
        Arg("o", OUT, "float32", (N,)),
    ],
    grid=(N // BLOCK,),
    reference=reference,
    make_inputs=make_inputs,
)
'''


def per_lane_bytes(block):
    """Bytes of one f32 tile each lane must hold, once the DMA scatters it."""
    return (block / LANES) * 4


# Triton itself rejects a non-power-of-2 `tl.arange` and any tensor over
# TRITON_MAX_TENSOR_NUMEL, at stage 1, before MLIR exists. Those are NOT this
# harness's constraints and no stage-3 pass can lift them, so probing them only
# measures Triton. Every probe below is a legal Triton block size; anything that
# fails therefore fails on a rule THIS harness invented.
#
# (label, BLOCK, grid count, what this probe is asking)
PROBES = [
    ("baseline, BLOCK == lanes",      LANES,      8,
     "the size the hardware natively wants"),
    ("BLOCK = 2 x lanes",             LANES * 2,  8,
     "a clean multiple -- should be fine either way"),
    ("BLOCK = lanes / 2",             LANES // 2, 8,
     "THE probe: power-of-2, legal Triton, fewer elements than lanes"),
    ("BLOCK = lanes / 4",             LANES // 4, 8,
     "same class, further from lane width"),
    ("BLOCK = 8",                     8,          8,
     "a tile far smaller than the machine -- must become a partial tile"),
    ("BLOCK = 1024 (8 x lanes)",      1024,       4,
     "large but still a clean multiple"),
]

# Capacity is not reachable from a 1-D probe: Triton caps a tensor at
# TRITON_MAX_TENSOR_NUMEL (1<<20) = 4 MB of f32, and spread over LANES that is
# 32 KB per lane, under SPAD_SIZE. It takes GEMM's THREE live tiles (A, B, acc)
# to exceed one lane's 64 KB, so the capacity question gets its own probe.
GEMM_PROBES = [
    ("gemm 128x128, 2 K-tiles",   128, 128, 128, 256,
     "the shape tests/cases/smoke/gemm.py ships -- baseline"),
    ("gemm 512x512 tiles",        512, 512, 512, 1024,
     "3 live tiles x 512x512 f32 -> far past one lane's spad"),
]


def run_probe(label, block, grid, question, stage, workdir, verbose):
    n = block * grid
    name = f"probe{block}_kernel"
    src = TEMPLATE.format(root=ROOT, n=n, block=block, name=name)
    path = os.path.join(workdir, f"{name}.py")
    with open(path, "w") as f:
        f.write(src)

    cmd = [sys.executable, os.path.join(ROOT, DRIVER), path, "--to", stage,
           "--workdir", workdir]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    ok = p.returncode == 0
    out = p.stdout + p.stderr
    return _classify(label, block, n, ok, out, per_lane_bytes(block), question)


# triton-compile is Triton's frontend and the linalg lowering: their rules are
# upstream's and a pass added to stage 3 cannot lift them. Everything else is ours.
FRONTEND_STAGES = {"triton-compile"}


def _classify(label, block, n, ok, out, per_lane, question):
    """Which stage rejected this, and is that stage ours to fix?"""
    stage = ""
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("resume") and "--from" in s:   # run.py prints the failed stage here
            stage = s.split("--from")[-1].strip().split()[0]
    reason = ""
    for line in out.splitlines():
        s = line.strip()
        if any(k in s for k in ("divisible", "lanes", "tile dim", "Error",
                                "error:", "UnsupportedIR", "Exception", "exceeds")):
            if s and not s.startswith("$"):
                reason = s[:130]
                break
    if ok:
        kind = "ok"
    elif stage in FRONTEND_STAGES:
        kind = "frontend"     # Triton's own limit -- not evidence about our tiling
    else:
        kind = "harness"      # OUR rule -- this is what a tiling pass must fix
    return {"label": label, "block": block, "n": n, "ok": ok, "kind": kind,
            "stage": stage, "reason": reason, "question": question,
            "per_lane": per_lane, "out": out}


GEMM_TEMPLATE = '''\
"""Generated GEMM probe -- see tools/tiling_invariant.py."""
import sys
sys.path.insert(0, {root!r})
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec, Arg, IN, OUT

M, N, K = {m}, {n}, {k}
BM, BN, BK = {bm}, {bn}, {bk}


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def {name}(a_ptr, b_ptr, c_ptr,
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
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    return {name}


def make_inputs(torch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {{
        "a": torch.randn(M, K, generator=g, dtype=torch.float32),
        "b": torch.randn(K, N, generator=g, dtype=torch.float32),
    }}


def reference(inputs):
    import torch
    return {{"c": torch.matmul(inputs["a"], inputs["b"])}}


SPEC = KernelSpec(
    name={name!r},
    kernel=kernel,
    signature={{"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "constexpr", "N": "constexpr", "K": "constexpr",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"}},
    constexprs={{"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK}},
    args=[
        Arg("a", IN, "float32", (M, K)),
        Arg("b", IN, "float32", (K, N)),
        Arg("c", OUT, "float32", (M, N)),
    ],
    grid=(M // BM, N // BN),
    reference=reference,
    rtol=1e-3, atol=1e-3,
    make_inputs=make_inputs,
)
'''


def run_gemm_probe(label, bm, bn, bk, k, question, stage, workdir):
    """Capacity probe: three live tiles must fit in ONE lane's scratchpad."""
    name = f"gemmprobe{bm}_kernel"
    src = GEMM_TEMPLATE.format(root=ROOT, m=bm * 2, n=bn * 2, k=k,
                               bm=bm, bn=bn, bk=bk, name=name)
    path = os.path.join(workdir, f"{name}.py")
    with open(path, "w") as f:
        f.write(src)
    cmd = [sys.executable, os.path.join(ROOT, DRIVER), path, "--to", stage,
           "--workdir", workdir]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    # A, B, acc live at once; the DMA splits dim 0 across lanes.
    live = (bm * bk + bk * bn + bm * bn) * 4 / LANES
    return _classify(label, bm, bm * bn, p.returncode == 0,
                     p.stdout + p.stderr, live, question)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numeric", action="store_true",
                    help="run all the way through Spike and compare against torch "
                         "(catches the silent-wrong-answer case)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    stage = "verify" if args.numeric else "transform"

    print(f"lanes      : {LANES}")
    print(f"spad/lane  : {machine.SPAD_SIZE} B")
    print(f"stage      : -> {stage}")
    print()
    print("ASSUMPTION: a kernel is one core's work and its tile FITS IN THE SPAD.")
    print("            The source picks the tile; the compiler splits it into units")
    print("            the hardware can issue (lanes / vector width / systolic).")
    print("            A size that is REFUSED by name is fine -- that is the")
    print("            source/compiler boundary. A size that compiles and is")
    print("            silently WRONG is the failure this looks for.")
    print()
    print(f"{'probe':<32} {'BLOCK':>10} {'B/lane':>9}  result")
    print("-" * 100)

    results = []
    with tempfile.TemporaryDirectory(prefix="tiling-probe-") as workdir:
        for label, block, grid, question in PROBES:
            r = run_probe(label, block, grid, question, stage, workdir,
                          args.verbose)
            results.append(r)
            _report(r, args.verbose)

        print()
        print(f"{'capacity probe (gemm)':<32} {'BM':>10} {'B/lane':>9}  result")
        print("-" * 100)
        for label, bm, bn, bk, k, question in GEMM_PROBES:
            r = run_gemm_probe(label, bm, bn, bk, k, question, stage, workdir)
            results.append(r)
            _report(r, args.verbose)

    print("-" * 100)
    print()

    harness = [r for r in results if r["kind"] == "harness"]
    frontend = [r for r in results if r["kind"] == "frontend"]

    if frontend:
        print(f"{len(frontend)} probe(s) stopped in triton-compile, NOT here:")
        for r in frontend:
            print(f"  - {r['label']} (stage {r['stage']})")
        print("  These are upstream limits. A pass added to stage 3 cannot lift")
        print("  them, so they are not evidence about this harness's tiling.")
        print()

    if harness:
        print(f"REFUSED BY THIS HARNESS: {len(harness)} legal-Triton size(s).")
        for r in harness:
            print(f"  * {r['label']:<30} BLOCK={r['block']} (stage {r['stage']})")
            if r["reason"]:
                print(f"      {r['reason']}")
        print()
        print("This is where the source/compiler boundary currently sits, not")
        print("necessarily a bug. Read each message: a size refused BY NAME is the")
        print("harness saying the source has to pick something the hardware can")
        print("express. It becomes a bug only if the limit named is one")
        print("vectorize / legalize could have split -- lanes, vector")
        print("width, or the systolic dimension -- splitting those is our job.")
        print()
        print("Capacity is deliberately NOT split (README Phase 5): a tile over")
        print("SPAD_SIZE is refused by spad.check_budget rather than streamed.")
        return 1

    print("EVERY size this harness gets to see compiles.")
    print("Sizes rejected upstream by Triton itself are listed above and are not")
    print("evidence about this harness.")
    return 0


def _report(r, verbose):
    status = {"ok": "compiled", "harness": "REJECTED (ours)",
              "frontend": "rejected (upstream)"}[r["kind"]]
    print(f"{r['label']:<32} {r['block']:>10} {r['per_lane']:>9.0f}  {status}")
    if r["reason"]:
        print(f"{'':<32} {'':>10} {'':>9}  -> {r['reason']}")
    if verbose and not r["ok"]:
        print("     " + "\n     ".join(r["out"].splitlines()[-15:]))


if __name__ == "__main__":
    sys.exit(main())
