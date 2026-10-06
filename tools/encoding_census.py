"""encoding_census: stage 3 one pass at a time over a sweep's 02h-layouts.mlir; after each pass, every
rank>=1 data tensor that lost its TritonGPU encoding, by pass and op, and what is left for p02.
Usage: python tools/encoding_census.py <sweep-dir> [-j 16] [--kernel GLOB]"""
import argparse
import collections
import fnmatch
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler import toolchain                       # noqa: E402
from pytorchsim_triton_compiler.pipeline import TTSHARED_PIPELINE, TTGIR_PRINT  # noqa: E402

OP = re.compile(r'"([a-z_]+\.[a-z_.]+)"')
TENSOR = re.compile(r'tensor<([^<>]*(?:<[^<>]*(?:<[^<>]*(?:<[^<>]*>[^<>]*)*>[^<>]*)*>[^<>]*)*)>')


def get_passes():
    """Every flag of the stage-3 pipeline as its own step, in order."""
    return [f for _, flags in TTSHARED_PIPELINE for f in flags]


def get_bare_results(path):
    """{op: count} of results that are rank>=1 tensors of data (not pointers, not shape
    vectors of a reshape) with no encoding."""
    got = collections.Counter()
    with open(path) as f:
        for line in f:
            m = OP.search(line)
            if not m or "->" not in line:
                continue
            res = line.rsplit("->", 1)[1]
            for t in TENSOR.finditer(res):
                if res[:t.start()].endswith("ptr<"):
                    continue
                body = t.group(1)
                head = body.split(",")[0]
                if "x" not in head or head[:1] in "fib" or "#ttg" in body or "ptr" in head:
                    continue
                if m.group(1) == "arith.constant" and re.fullmatch(r"\d+xi64", head):
                    continue
                got[m.group(1)] += 1
    return got


def run_kernel(src, tmp):
    """[(pass, {op: new bare results})], and the bare results left at the end; or an error."""
    steps, flags, prev = [], [], collections.Counter()
    for i, flag in enumerate(get_passes()):
        flags.append(flag)
        out = os.path.join(tmp, "%02d.mlir" % i)
        r = subprocess.run([toolchain.TRITON_SHARED_OPT, *flags, *TTGIR_PRINT, src, "-o", out],
                           capture_output=True, text=True)
        if r.returncode:
            return None, flag
        cur = get_bare_results(out)
        steps.append((flag, {op: n - prev.get(op, 0) for op, n in cur.items() if n > prev.get(op, 0)}))
        prev = cur
    return steps, prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("-j", type=int, default=16)
    ap.add_argument("--kernel", default="*")
    a = ap.parse_args()
    kernels = [k for k in sorted(os.listdir(a.sweep)) if fnmatch.fnmatch(k, a.kernel)
               and os.path.exists(os.path.join(a.sweep, k, "02h-layouts.mlir"))]

    def one(k):
        with tempfile.TemporaryDirectory() as tmp:
            return k, run_kernel(os.path.join(a.sweep, k, "02h-layouts.mlir"), tmp)

    made, where, left, failed = collections.Counter(), collections.defaultdict(set), collections.Counter(), []
    with ThreadPoolExecutor(a.j) as ex:
        for k, (steps, end) in ex.map(one, kernels):
            if steps is None:
                failed.append((k, end))
                continue
            for flag, ops in steps:
                for op, n in ops.items():
                    made[(flag, op)] += n
                    where[(flag, op)].add(k)
            for op, n in end.items():
                left[op] += n
    print("%d kernels, %d stopped in stage 3 (refusals): %s" % (
        len(kernels), len(failed), ", ".join("%s@%s" % (k[14:], f.split("=")[0]) for k, f in failed)))
    print("\n%-48s %-30s %7s %7s  %s" % ("pass that made it", "op", "results", "kernels", "example"))
    for key in sorted(made, key=lambda x: (get_passes().index(x[0]), -len(where[x]))):
        print("%-48s %-30s %7d %7d  %s" % (key[0][:48], key[1], made[key], len(where[key]),
                                         sorted(where[key])[0][14:]))
    print("\nleft in 03a-ttshared.mlir (what p02 reads):")
    for op, n in left.most_common():
        print("  %-30s %5d" % (op, n))


if __name__ == "__main__":
    main()
