"""The binary stage counts vector spills off llc's regalloc remarks (NumVectorSpills/NumVectorReloads, a
function's last allocation run), not off the assembly, in total and by location; toolchain.LLC must report them.
    python3 tests/runtime/vector_spill_remark_cases.py        exit 0 iff every case holds"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402
from pytorchsim_triton_compiler.lowering.elf import build  # noqa: E402

REMARK = """--- !Missed
Pass:            regalloc
Name:            SpillReloadCopies
Function:        {fn}
Args:
{args}  - String:          generated in function
...
"""

VECTOR_SPILL = """define void @f(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
  %v = load <8 x float>, ptr addrspace(1) %src, align 32
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  store <8 x float> %v, ptr addrspace(1) %dst, align 32
  ret void
}
"""

# VECTOR_SPILL with every instruction at line 3 of the loops' file: its spill and reload name loop 3.
LOCATED = VECTOR_SPILL.replace(") {\n", ") !dbg !3 {\n", 1)
LOCATED = "\n".join(l + (", !dbg !4" if l.startswith("  ") else "") for l in LOCATED.split("\n")) + """
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "torchsim-loop", directory: "")
!2 = !DISubroutineType(types: !{})
!3 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DILocation(line: 3, column: 0, scope: !3)
!5 = !{i32 2, !"Debug Info Version", i32 3}
"""


def _remarks(*remarks):
    d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    path = os.path.join(d, "r.yaml")
    with open(path, "w") as f:
        for fn, args in remarks:
            f.write(REMARK.format(fn=fn, args="".join(f"  - {k}: '{v}'\n" for k, v in args.items())))
    return path


def _a_function_counts_once_by_its_last_remark():
    """The vector allocation and the scalar one each report the whole function: summed, a vector
    spill would count twice."""
    got = build.read_places(_remarks(
        ("k", {"NumSpills": 2, "NumReloads": 2, "NumVectorSpills": 2, "NumVectorReloads": 2}),
        ("k", {"NumSpills": 15, "NumReloads": 22, "NumVectorSpills": 2, "NumVectorReloads": 2}),
        ("g", {"NumSpills": 1, "NumVectorSpills": 1})))[:2]
    return got == (3, 2), got


def _scalar_spills_are_no_vector_spills():
    got = build.read_places(_remarks(("k", {"NumSpills": 8, "NumReloads": 8})))[:2]
    return got == (0, 0), got


def _no_remarks_file_no_spills():
    got = build.read_places("/nonexistent/r.yaml")[:2]
    return got == (0, 0), got


def _llc(ir):
    toolchain.ensure_bindings()
    t = machine.target()
    with tempfile.TemporaryDirectory() as d:
        ll, yml = os.path.join(d, "k.ll"), os.path.join(d, "k.yaml")
        with open(ll, "w") as f:
            f.write(ir)
        subprocess.run([toolchain.LLC, f"-march={t.llc_march()}", "-O3", f"-mattr={t.llc_mattr()}",
                        ll, "-o", os.devnull, f"-pass-remarks-output={yml}",
                        "-pass-remarks-filter=regalloc"], check=True, capture_output=True)
        return build.read_places(yml)


def _llc_reports_a_vector_spill():
    """FAILs on an llc whose regalloc remark has no NumVectorSpills: every spill would read as none."""
    got = _llc(VECTOR_SPILL)[:2]
    return got == (1, 1), {"llc": toolchain.LLC, "got": got}


def _llc_places_a_vector_spill_and_its_units():
    """FAILs on an llc whose per-line remarks carry no NumVectorSpills or VectorSpilledRegisterUnits:
    the retry would name no place, or size one by no units."""
    got = _llc(LOCATED)
    ok = got.loops == {3: 2} and got.dots == {} and got.outside == 0 and got.units.get(("loop", 3), 0) > 0
    return ok, {"llc": toolchain.LLC, "got": repr(got)}


CASES = {"a_function_counts_once_by_its_last_remark": _a_function_counts_once_by_its_last_remark,
         "scalar_spills_are_no_vector_spills": _scalar_spills_are_no_vector_spills,
         "no_remarks_file_no_spills": _no_remarks_file_no_spills,
         "llc_reports_a_vector_spill": _llc_reports_a_vector_spill,
         "llc_places_a_vector_spill_and_its_units": _llc_places_a_vector_spill_and_its_units}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:42} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:400]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
