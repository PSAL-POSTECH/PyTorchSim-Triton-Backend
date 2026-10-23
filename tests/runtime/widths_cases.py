"""One spill decision per kernel: the widths stamp read back, two builds at other widths refused by name,
and a kernel whose runtime attribute records once moved the decision built at the widths it has without them.
    python3 tests/runtime/widths_cases.py        exit 0 iff every case gets what it names"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from pytorchsim_triton_compiler.lowering import widths  # noqa: E402

D = widths.decision(8, {7: 4}, {1: 2})


def _module(d, stamped=True):
    """A module file in `d`, stamped with D."""
    from mlir.ir import Context, Location, Module
    path = os.path.join(d, "m.mlir")
    with Context(), Location.unknown():
        m = Module.parse("module {}")
        if stamped:
            widths.stamp(m, lmul=D["lmul"], loop_lmul=D["loop_lmul"])
            widths.stamp(m, dot_lmul=D["dot_lmul"])
        open(path, "w").write(str(m))
    return path


def a_stamp_reads_back_as_the_decision(d):
    """Breaks if transform's and lower's halves of the stamp do not merge into the decision they state."""
    got = widths.read(_module(d))
    return got == D == {"lmul": 8, "loop_lmul": {"7": 4}, "dot_lmul": {"1": 2}}, got


def builds_at_other_widths_are_refused(d):
    """Breaks if two builds whose loop or stream LMULs differ, or one that states none, pass as one
    decision: the Spike ELF and the cycle binary would run different tiles."""
    said = []
    for other in (widths.decision(8, {7: 8}, {1: 2}), widths.decision(8, {7: 4}, {}), None):
        try:
            widths.refuse_unless_same({"the Spike ELF": D, "the cycle binary": other})
            said.append("no refusal")
        except widths.WidthsDiffer as e:
            said.append(str(e))
    ok = ("the cycle binary is built at widths" in said[0] and "one spill decision" in said[0]
          and "the cycle binary is built at widths" in said[1]
          and "the cycle binary states no torchsim.widths" in said[2])
    return ok and widths.refuse_unless_same({"a": D, "b": json.loads(json.dumps(D))}) == D, said


PAGED = "tests/cases/mechanism/rtattr/rtattr_paged_block_table_tog.py"


def records_do_not_move_the_widths(d):
    """Breaks if the record calls decide the widths: rtattr_paged_block_table's ELF with records spills
    around them, and judged on it the retry narrowed loop 7 to LMUL 4 (vector<32xf32>, 4933 TOGSim
    cycles against main's 5148); judged without them, every f32 vector is LMUL 8's 64 lanes, as on main."""
    from mlir.ir import Context, Module, VectorType
    work = os.path.join(d, "paged")
    got = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"),
                          os.path.join(ROOT, PAGED), "--workdir", work, "--to", "binary"],
                         cwd=ROOT, capture_output=True, text=True)
    if got.returncode != 0:
        return False, (got.stdout + got.stderr)[-600:]
    shapes = set()
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(open(os.path.join(work, "08-transformed.mlir")).read())
        stack = [module.operation]
        while stack:
            op = stack.pop()
            shapes |= {tuple(VectorType(r.type).shape) for r in op.results if isinstance(r.type, VectorType)}
            stack.extend(o for region in op.regions for block in region.blocks for o in block.operations)
    stated = json.load(open(os.path.join(work, "kernel.json"))).get("widths")
    want = widths.decision(8)
    return shapes == {(64,)} and stated == want, {"vector shapes": sorted(shapes), "widths": stated}


CASES = [a_stamp_reads_back_as_the_decision, builds_at_other_widths_are_refused, records_do_not_move_the_widths]


def main():
    failed = 0
    for case in CASES:
        with tempfile.TemporaryDirectory() as d:
            try:
                ok, got = case(d)
            except Exception as e:  # noqa: BLE001
                ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {case.__name__}  {json.dumps(got, default=str)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
