"""build.read_places on regalloc remarks as llc writes them for RISC-V: each allocation run (vector registers,
then scalar ones) reports the whole function and ends in its SpillReloadCopies. The last run is read, by the
vector keys alone (NumVectorSpills/NumVectorReloads, VectorSpilledRegisterUnits): scalar spills name no place.
    python3 tests/passes/spill_places_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.codegen import build  # noqa: E402


def _remark(name, file, line, spills, reloads, units=0, vspills=0, vreloads=0, vunits=0):
    loc = f"DebugLoc: {{ File: '{file}', Line: {line}, Column: 0 }}\n" if file else ""
    vec = "".join(f"  - {k}: '{v}'\n" for k, v in (("NumVectorSpills", vspills),
                                                   ("VectorSpilledRegisterUnits", vunits),
                                                   ("NumVectorReloads", vreloads)) if v)
    return (f"--- !Missed\nPass: regalloc\nName: {name}\n{loc}Function: f\nArgs:\n"
            f"  - NumSpills: '{spills}'\n  - SpilledRegisterUnits: '{units}'\n"
            f"  - NumReloads: '{reloads}'\n{vec}...\n")


def _read(text):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
    try:
        return build.read_places(f.name)
    finally:
        os.unlink(f.name)


def the_last_run_names_places_by_their_vector_spills():
    """Breaks if both runs are summed (a vector spill counted twice) or a scalar spill names a place."""
    vector_run = (_remark("BlockSpillReloadCopies", "pytorchsim-dot", 3, 2, 1, 8, 2, 1, 8)
                  + _remark("BlockSpillReloadCopies", "pytorchsim-loop", 5, 1, 1, 2, 1, 1, 2)
                  + _remark("SpillReloadCopies", None, 0, 4, 3, 12, 4, 3, 12))
    scalar_run = (_remark("BlockSpillReloadCopies", "pytorchsim-dot", 3, 52, 91, 58, 2, 1, 8)
                  + _remark("BlockSpillReloadCopies", "pytorchsim-loop", 5, 1, 1, 2, 1, 1, 2)
                  + _remark("BlockSpillReloadCopies", "pytorchsim-loop", 9, 7, 7, 7)
                  + _remark("SpillReloadCopies", None, 0, 61, 99, 69, 4, 3, 12))
    got = _read(vector_run + scalar_run)
    assert (got.spills, got.reloads) == (4, 3), got
    assert (got.dots, got.loops) == ({3: 3}, {5: 2}), got
    assert got.units == {("dot", 3): 8, ("loop", 5): 2}, got.units
    assert got.outside == 2, got.outside      # 7 vector spills and reloads, 5 of them named


def a_function_whose_vector_run_reports_nothing_names_no_place():
    """Breaks if the first SpillReloadCopies is taken for the vector run: with no vector spill and no
    copy that run writes no remark, and the scalar run's spills would name loop 5."""
    got = _read(_remark("BlockSpillReloadCopies", "pytorchsim-loop", 5, 3, 3, 3)
                + _remark("SpillReloadCopies", None, 0, 3, 3, 3))
    assert got == (0, 0, {}, {}, 0, {}), got


CASES = [the_last_run_names_places_by_their_vector_spills,
         a_function_whose_vector_run_reports_nothing_names_no_place]


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
