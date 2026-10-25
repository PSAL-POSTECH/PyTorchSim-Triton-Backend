"""pytorchsim-lower-gather puts a lane gather's pops into the loop that reads its
destination a register group of rows a trip, the pushes before that loop: each trip pops its own rows,
as the v6e pops a vperm where its value is used. Read off the pass's own output.
    python3 tests/passes/xlu_gather_consumer_pop_cases.py   exit 0 iff all hold"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.tpu_kind import lib_xlu  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import trip_count, walk  # noqa: E402

KERNEL = os.path.join(ROOT, "tests", "cases", "mechanism", "ops", "ops_xlu_gather_rows_consumed.py")


def _loop_of(o):
    """The innermost loop `o` sits in, or None."""
    p = o.parent
    while p is not None and p.name not in ("affine.for", "scf.for"):
        p = p.parent
    return p


def _placed():
    """(the loops holding the gather's pops, with their trip counts; the loops holding its pushes)."""
    with tempfile.TemporaryDirectory() as d:
        w = os.path.join(d, "w")
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), KERNEL,
                        "--workdir", w, "--to", "lower"], check=False, capture_output=True)
        out = [f for f in sorted(os.listdir(os.path.join(w, "09-passes"))) if "lower-gather" in f]
        with ir.Context() as ctx, ir.Location.unknown():
            ctx.allow_unregistered_dialects = True
            m = ir.Module.parse(open(os.path.join(w, "09-passes", out[0])).read())
            pops, pushes = [], []
            for o in walk(m.operation):
                if o.name == "vcix.iv" and ir.IntegerAttr(o.attributes["opcode"]).value == lib_xlu.XLU_PUSH_OPCODE:
                    pushes.append(_loop_of(o))
                elif o.name == "vcix.v.i" and ir.IntegerAttr(o.attributes["opcode"]).value == lib_xlu.XLU_POP_OPCODE:
                    pops.append(_loop_of(o))
            return [(p, trip_count(p) if p is not None else None) for p in pops], pushes


def the_gathers_pops_sit_in_the_loop_that_reads_its_rows():
    """Breaks if the pops go back to right after the pushes, into a destination the loop reads again."""
    pops, pushes = _placed()
    assert pops and all(p is not None and trips and trips > 1 for p, trips in pops), pops
    held = [p.operation for p, _ in pops]
    assert pushes and not any(p is not None and p.operation in held for p in pushes), pushes


CASES = [the_gathers_pops_sit_in_the_loop_that_reads_its_rows]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
