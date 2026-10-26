"""The tpu kind's answers on the layout port (ext/tpu_kind/layout_model.py), with no IR loaded: the element
pairs two layouts imply, and a conversion's cost -- a copy in each bank where no element leaves its lane,
the XLU move lower-convert would make otherwise.
    python3 tests/passes/layout_port_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
from pytorchsim_triton_compiler.lowering.ext.tpu_kind import layout_model  # noqa: E402
from pytorchsim_triton_compiler.lowering.layout_port import Layout  # noqa: E402

_ZERO = ((0, 0),) * 4
# 2x8: eight lanes hold two elements each (ops_shape's %38 source), and two lanes hold a row each
_PAIRS = Layout((2, 8), 32, ((0, 1),), ((0, 2), (0, 4), (1, 0)) + _ZERO)
_ROWS = Layout((2, 8), 32, ((0, 1), (0, 2), (0, 4)), ((1, 0),) + (((0, 0),) * 6))


def no_ir_is_loaded():
    """The answers need no MLIR bindings: the TTGIR passes' process holds Triton's own, and a second
    LLVM there refuses to load. Breaks if layout_model imports a module that loads mlir."""
    assert "mlir" not in sys.modules, sorted(m for m in sys.modules if m.startswith("mlir"))


def each_destination_slot_takes_the_slot_holding_its_element():
    """Lane 1, register 1 of _ROWS holds element (1, 1), which sits in lane 4, register 1, of _PAIRS.
    Breaks if a coordinate is read off the bases wrongly."""
    pairs = dict((d, s) for s, d in layout_model.pairs_of_layouts(_PAIRS, _ROWS))
    assert pairs[(1, 1)] == (4, 1), pairs[(1, 1)]


def a_conversion_that_keeps_every_lane_is_a_bank_copy():
    """The same layout in both, its rows held in two lanes and the other lanes repeating them: one
    register's rows a cycle. Breaks if a lane that holds its element takes it from another lane."""
    assert layout_model.convert_cost(_ROWS, _ROWS) == 1.0, layout_model.convert_cost(_ROWS, _ROWS)


def a_conversion_across_lanes_is_the_move_lower_convert_makes():
    """_PAIRS into _ROWS crosses lanes: a finite count of the unit's cycles, more than its latency.
    Breaks if no XLU op is found for it, or the move is priced as a copy."""
    cost = layout_model.convert_cost(_PAIRS, _ROWS)
    assert 123 < cost < float("inf"), cost


def a_part_axis_on_a_lane_bit_is_not_lane_local():
    """An [8, 2] whose axis of 2 is a lane bit: a split's halves lie in other lanes than its results want,
    and a join's operands, held in the first lanes only, would have to cross to theirs; with that axis in
    registers each lane keeps its own. Breaks if either is called lane-local on a lane bit."""
    lanes_on_axis = Layout((8, 2), 32, ((1, 0),), ((2, 0), (4, 0), (0, 1)) + _ZERO)
    regs_on_axis = Layout((8, 2), 32, ((0, 1),), ((1, 0), (2, 0), (4, 0)) + _ZERO)
    half = Layout((8,), 32, (), ((1,), (2,), (4,)) + ((0,),) * 4)
    assert not layout_model.layout_legal("split", lanes_on_axis, half)
    assert not layout_model.layout_legal("join", half, lanes_on_axis)
    assert layout_model.layout_legal("split", regs_on_axis, half)
    assert layout_model.layout_legal("join", half, regs_on_axis)


CASES = [no_ir_is_loaded, each_destination_slot_takes_the_slot_holding_its_element,
         a_conversion_that_keeps_every_lane_is_a_bank_copy,
         a_conversion_across_lanes_is_the_move_lower_convert_makes,
         a_part_axis_on_a_lane_bit_is_not_lane_local]


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
