"""materialize_layouts._divides: a runtime DMA bound on a dim cut into digits is a box on its outer digit when the
inner run divides it, read from the bound's producers. The bound is convnextv2's masked tail, built as Inductor's
`xindex < xnumel` lowers it: min(max(min(pid*8192 + 8192, xnumel), pid*8192) - pid*8192, 8192).
    python3 tests/passes/clamp_divisor_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from pytorchsim_triton_compiler.passes.transforms import materialize_layouts  # noqa: E402

_BOUND = """
module {{
  func.func @f(%pid: i32) -> index {{
    %c8192_i32 = arith.constant 8192 : i32
    %c8192 = arith.constant 8192 : index
    %cn = arith.constant {xnumel} : index
    %0 = arith.muli %pid, %c8192_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.addi %1, %c8192 : index
    %3 = arith.minsi %2, %cn : index
    %4 = arith.maxsi %3, %1 : index
    %5 = arith.subi %4, %1 : index
    %6 = arith.minsi %5, %c8192 : index
    return %6 : index
  }}
}}
"""


#: That bound divided by a constant, as PtrAnalysis writes it where it cuts a wrap walked in whole laps:
#: swinv2's shifted-window attention reaches materialize_layouts with its row bound behind an arith.divsi.
_QUOTIENT = _BOUND.replace("""    %6 = arith.minsi %5, %c8192 : index
    return %6 : index""", """    %6 = arith.minsi %5, %c8192 : index
    %cd = arith.constant {by} : index
    %7 = arith.divsi %6, %cd : index
    return %7 : index""")


def _parse(text):
    """The value the module's function returns."""
    from mlir.ir import Module
    m = Module.parse(text)
    ret = m.body.operations[0].regions[0].blocks[0].operations[-1]
    return m, ret.operands[0]


def _bound(xnumel):
    """The value the function returns: the tail bound of a block of 8192 over `xnumel` elements."""
    return _parse(_BOUND.format(xnumel=xnumel))


def _quotient(xnumel, by):
    """That tail bound divided by `by`, the shape PtrAnalysis leaves on a cut wrap."""
    return _parse(_QUOTIENT.format(xnumel=xnumel, by=by))


def a_tail_bound_over_a_multiple_of_the_lanes_is_a_multiple_of_them():
    """Breaks if the bound stops being read through its producers: the cut dim's mask is refused again."""
    m, v = _bound(6272)
    assert materialize_layouts._divides(128, v)


def a_tail_bound_over_any_other_count_is_not():
    """Breaks if a bound the lanes do not divide is taken as a box on the outer digit, which drops elements."""
    m, v = _bound(6200)
    assert not materialize_layouts._divides(128, v)


def an_exactly_divided_bound_keeps_what_the_division_leaves():
    """Breaks if a quotient stops being read: swinv2's shifted-window DMA is refused on its [64, 2] cut,
    whose bound is this one over 64 and is always even."""
    m, v = _quotient(12288, 64)
    assert materialize_layouts._divides(2, v)


def a_division_that_is_not_exact_keeps_nothing():
    """Breaks if a quotient is taken for more than it is: 4096 over 3 leaves no multiple to stand on."""
    m, v = _quotient(12288, 3)
    assert not materialize_layouts._divides(2, v)


CASES = [a_tail_bound_over_a_multiple_of_the_lanes_is_a_multiple_of_them,
         a_tail_bound_over_any_other_count_is_not,
         an_exactly_divided_bound_keeps_what_the_division_leaves,
         a_division_that_is_not_exact_keeps_nothing]


def main():
    from mlir.ir import Context, Location
    bad = 0
    for case in CASES:
        with Context(), Location.unknown():
            try:
                case()
                print(f"ok    {case.__name__}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
