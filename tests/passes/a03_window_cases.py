"""a03 cuts a window of a tile whose axis its layout splits into digits: one inside the innermost digit,
or one that covers whole innermost digits from their boundary (a cat's half, laid with its half bit
outermost); any other is refused.
    python3 tests/passes/a03_window_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from pytorchsim_triton_compiler.lowering.passes.a03_materialize_layouts import _inner_digits  # noqa: E402


def a_half_covering_whole_inner_digits_is_cut_on_them():
    """16 as digits [2, 2, 4]: [0, 8) and [8, 16) are the outer digit's two values over the inner [2, 4].
    Breaks if a window that covers whole inner digits is refused."""
    assert _inner_digits([2, 2, 4], 0, 8) == [2, 4]
    assert _inner_digits([2, 2, 4], 8, 8) == [2, 4]


def a_window_off_the_boundary_is_not():
    """[4, 12) covers eight elements but starts inside a digit: none. Breaks if a window that splits a
    digit is taken as whole ones."""
    assert _inner_digits([2, 2, 4], 4, 8) == []
    assert _inner_digits([2, 2, 4], 0, 4) == []


CASES = [a_half_covering_whole_inner_digits_is_cut_on_them, a_window_off_the_boundary_is_not]


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
