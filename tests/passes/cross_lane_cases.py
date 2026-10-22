"""A lane axis longer than the machine folds onto its lanes lane-full by lane-full, a last partial
lane-full as deep as a whole one -- in the bank's reserve and in the XLU's rounds alike.
    python3 tests/passes/cross_lane_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import (  # noqa: E402
    lane_fulls, need_per_lane)
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e.lib_xlu import (  # noqa: E402
    XLU_PERMUTE, choose_xlu_op, emitter_of, fold_onto_lanes, gather_rounds)

LANES = 32


def _window_pairs(n, offset, into):
    """The (from, to) pairs of a `n`-wide tile copied into (or cut out of) a longer one at
    `offset`, as lower_copies builds them before the fold: (lane axis, in-bank)."""
    pairs = [((i, 0), (offset + i, 0)) for i in range(n)]
    return pairs if into else [(t, f) for f, t in pairs]


def a_partial_lane_full_is_as_deep_as_a_whole_one():
    assert lane_fulls(48, LANES) == 2 and lane_fulls(64, LANES) == 2 and lane_fulls(32, LANES) == 1
    assert need_per_lane(48, 48, LANES) == 2
    assert need_per_lane(48 * 4, 48, LANES) == 8, "48 x 4 over 32 lanes: two lane-fulls, 4 deep"
    assert need_per_lane(64 * 4, 64, LANES) == 8
    assert need_per_lane(32 * 4, 32, LANES) == 4
    assert need_per_lane(12 * 16, 12, 8) == 32


def the_fold_puts_the_second_lane_full_one_row_deeper():
    got = fold_onto_lanes(_window_pairs(32, 16, True), [32], [48], LANES)
    assert got[0] == ((0, 0), (16, 0)) and got[15] == ((15, 0), (31, 0))
    assert got[16] == ((16, 0), (0, 1)) and got[31] == ((31, 0), (15, 1))


def the_fold_keeps_the_rows_of_a_deeper_tile_together():
    pairs = [((i, r), (16 + i, r)) for i in range(32) for r in range(4)]
    got = dict(fold_onto_lanes(pairs, [32, 4], [48, 4], LANES))
    assert got[(0, 3)] == (16, 3) and got[(16, 0)] == (0, 4) and got[(31, 3)] == (15, 7)


def a_shift_into_the_padded_tile_is_a_permute_of_two_rounds():
    xlu_op, plan = choose_xlu_op(_window_pairs(32, 16, True), [32], [48], LANES, False, 32,
                                 kinds=(XLU_PERMUTE,))
    assert emitter_of(xlu_op) == XLU_PERMUTE
    rows = sorted((at, depth, lo, hi) for at, _m, _a, depth, lo, hi, _d, _s in plan)
    assert rows == [(0, 0, 16, 32), (0, 1, 0, 16)], rows


def a_shift_out_of_the_padded_tile_reads_both_rows():
    xlu_op, plan = choose_xlu_op(_window_pairs(32, 15, False), [48], [32], LANES, False, 48)
    assert emitter_of(xlu_op) == XLU_PERMUTE
    rows = sorted((at, depth, lo, hi) for at, _m, _a, depth, lo, hi, _d, _s in plan)
    assert rows == [(0, 0, 0, 17), (1, 0, 17, 32)], rows


def a_tile_within_the_lanes_is_not_folded():
    got = fold_onto_lanes(_window_pairs(8, 4, True), [8], [12], 256)
    assert got == _window_pairs(8, 4, True)


def a_gather_into_a_destination_of_two_laps_passes_over_the_first_only():
    """Breaks if a lapping DISTRIBUTED destination is refused or gathered lap by lap: a [128] vector
    onto [256, 128] is one pass, and the lowering copies lap 0 into lap 1 in-lane."""
    pairs = [((j, 0), (i, j)) for i in range(2 * LANES) for j in range(LANES)]
    got = gather_rounds(pairs, [LANES], [2 * LANES, LANES], LANES, False)
    assert got == [(0, 0, 1)], got


CASES = [a_partial_lane_full_is_as_deep_as_a_whole_one,
         the_fold_puts_the_second_lane_full_one_row_deeper,
         the_fold_keeps_the_rows_of_a_deeper_tile_together,
         a_shift_into_the_padded_tile_is_a_permute_of_two_rounds,
         a_shift_out_of_the_padded_tile_reads_both_rows,
         a_tile_within_the_lanes_is_not_folded,
         a_gather_into_a_destination_of_two_laps_passes_over_the_first_only]


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
