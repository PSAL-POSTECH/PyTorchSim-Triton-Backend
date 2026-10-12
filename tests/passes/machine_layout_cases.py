"""A layout the machine cannot hold as TritonGPU wrote it is laid as the nearest one it can: the
lanes on one dim, that dim's lowest bits on the lanes in order, its wrapped bits the slowest
registers, every other dim's bits between -- and physical_perm accepts what machine_layout answers.
    python3 tests/passes/machine_layout_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.passes.lib_linear_layout import (  # noqa: E402
    machine_layout, physical_perm)


def _perm(bases, shape):
    got = physical_perm(bases, shape)
    assert isinstance(got, str), got
    return got, physical_perm(machine_layout(bases), shape)


def a_dim_split_between_the_lanes_and_the_registers_at_32_lanes():
    """qwen's sort_with_index tile on 32 lanes: dim 1 has four bits on the lanes and its fifth on a
    register, dim 2 takes a lane bit."""
    regs = [[0, 16, 0], [1, 0, 0]]
    lanes = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0]]
    why, got = _perm((regs, lanes), [2, 32, 2])
    assert why == "lanes over several dims", why
    assert got == [1, 2, 0], got
    new_regs, new_lanes = machine_layout((regs, lanes))
    assert new_lanes == [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], new_lanes
    assert new_regs == [[1, 0, 0], [0, 0, 1]], new_regs
    # before machine_layout, the lanes moved onto dim 1 left its fifth bit on a register:
    assert physical_perm(([[0, 16, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 0]] + lanes[1:]),
                         [2, 32, 2]) == "a dim split between the lanes and the registers"


def the_same_split_at_8_lanes():
    regs = [[0, 4, 0], [1, 0, 0]]
    lanes = [[0, 0, 1], [0, 1, 0], [0, 2, 0]]
    why, got = _perm((regs, lanes), [2, 8, 2])
    assert why == "lanes over several dims", why
    assert got == [1, 2, 0], got


def a_lane_dim_wrapped_ahead_of_another_dim():
    """The sort's [2, 512] tile on 256 lanes: dim 1 fills the lanes and its ninth bit is the FASTEST
    register, dim 0 the slowest; the machine wants the lane-full slowest."""
    regs = [[0, 256], [1, 0]]
    lanes = [[0, 1 << b] for b in range(8)]
    why, got = _perm((regs, lanes), [2, 512])
    assert why == "a dim split between the lanes and the registers", why
    assert got == [1, 0], got
    new_regs, new_lanes = machine_layout((regs, lanes))
    assert new_regs == [[1, 0], [0, 256]] and new_lanes == lanes, (new_regs, new_lanes)


def lanes_over_several_dims_still_land_on_the_fuller_dim():
    regs = [[0, 0, 2]]
    lanes = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [1, 0, 0]]
    why, got = _perm((regs, lanes), [2, 4, 4])
    assert why == "lanes over several dims", why
    new_regs, new_lanes = machine_layout((regs, lanes))
    assert new_lanes == [[0, 1, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]], new_lanes
    assert new_regs == [[0, 0, 1], [0, 0, 2], [1, 0, 0]], new_regs
    assert got == [1, 0, 2], got


def a_layout_the_machine_holds_is_left_alone():
    regs = [[1, 0], [2, 0]]
    lanes = [[0, 1], [0, 2], [0, 4]]
    assert machine_layout((regs, lanes)) == (regs, lanes)


def a_basis_stepping_two_bits_has_no_machine_layout():
    regs = [[0, 3]]
    lanes = [[0, 1], [1, 0]]
    assert machine_layout((regs, lanes)) == (regs, lanes)
    assert physical_perm((regs, lanes), [2, 8]) == "a basis that steps more than one bit"


CASES = [a_dim_split_between_the_lanes_and_the_registers_at_32_lanes, the_same_split_at_8_lanes,
         a_lane_dim_wrapped_ahead_of_another_dim, lanes_over_several_dims_still_land_on_the_fuller_dim,
         a_layout_the_machine_holds_is_left_alone, a_basis_stepping_two_bits_has_no_machine_layout]


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
