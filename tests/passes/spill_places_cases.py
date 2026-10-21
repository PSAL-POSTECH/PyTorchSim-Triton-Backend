"""build.read_places on regalloc remarks as llc writes them for RISC-V: a function's vector registers'
allocation first, then its scalar ones', each closed by the function's SpillReloadCopies. Only the
vector run names places, by the loop and matmul locations; what it spilled counts in register units.
    python3 tests/passes/spill_places_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.elf import build  # noqa: E402


def _remark(name, file, line, spills, reloads, units=0):
    loc = f"DebugLoc: {{ File: '{file}', Line: {line}, Column: 0 }}\n" if file else ""
    return (f"--- !Missed\nPass: regalloc\nName: {name}\n{loc}Function: f\nArgs:\n"
            f"  - NumSpills: '{spills}'\n  - SpilledRegisterUnits: '{units}'\n"
            f"  - NumReloads: '{reloads}'\n...\n")


def _read(text):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
    try:
        return build.read_places(f.name)
    finally:
        os.unlink(f.name)


def the_vector_run_names_places_and_the_scalar_run_does_not():
    text = (_remark("BlockSpillReloadCopies", "torchsim-dot", 3, 2, 1, 8)
            + _remark("BlockSpillReloadCopies", "torchsim-loop", 5, 1, 1, 2)
            + _remark("SpillReloadCopies", None, 0, 4, 3, 12)
            + _remark("BlockSpillReloadCopies", "torchsim-dot", 3, 50, 90, 50)
            + _remark("SpillReloadCopies", None, 0, 50, 90, 50))
    loops, dots, outside, units = _read(text)
    assert (dots, loops) == ({3: 3}, {5: 2}), (dots, loops)
    assert units == {("dot", 3): 8, ("loop", 5): 2}, units
    assert outside == 2, outside      # 7 in the vector run, 5 of them named


CASES = [the_vector_run_names_places_and_the_scalar_run_does_not]


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
