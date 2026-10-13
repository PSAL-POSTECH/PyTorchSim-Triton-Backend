"""The launch grid is part of what a kernel object was compiled for: the range it gives each
program id, and the refusal of a launch over any other grid.
    python3 tests/runtime/grid_contract_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract import kernel_object  # noqa: E402


def a_program_id_ranges_over_its_axis_of_the_grid():
    params = kernel_object.TRAILING_GRID_PARAMS
    for grid, want in (((8,), (7, 0, 0)), ((2, 1, 5), (1, 0, 4)), ((), (0, 0, 0))):
        got = kernel_object.program_id_ranges(grid)
        named = {params[len(params) + at]: r for at, r in got.items()}
        assert named == {"pidX": (0, want[0]), "pidY": (0, want[1]), "pidZ": (0, want[2])}, named


def the_grid_it_was_compiled_for_launches():
    manifest = {"name": "k", "grid": [8]}
    kernel_object.refuse_another_grid(manifest, (8,))
    kernel_object.refuse_another_grid(manifest, (8, 1, 1))


def another_grid_is_refused_and_both_are_named():
    manifest = {"name": "k", "grid": [8]}
    for other in ((9,), (4,), (8, 2)):
        try:
            kernel_object.refuse_another_grid(manifest, other)
        except kernel_object.ObjectError as e:
            assert "(8, 1, 1)" in str(e) and str(kernel_object.grid3(other)) in str(e), e
        else:
            raise AssertionError(f"grid {other} was not refused")


CASES = [a_program_id_ranges_over_its_axis_of_the_grid, the_grid_it_was_compiled_for_launches,
         another_grid_is_refused_and_both_are_named]


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
