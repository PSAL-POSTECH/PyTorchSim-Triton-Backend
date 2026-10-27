"""compose_reshape: a dim a reshape keeps whole goes through at its power-of-two extent, and a dim not a
power of two that the reshape merges or splits is still refused. The bases are yolov5s's maxpool tile.
    python3 tests/passes/compose_reshape_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from pytorchsim_triton_compiler.passes.utils.linear_layout import compose_reshape  # noqa: E402

#: tensor<16384x31> sliced from the 16384x32 tile Inductor's maxpool with indices reduces: its 16384
#: rows on 7 lane bits and 7 register bits, its 32 columns on the 5 register bits after them.
_REGS = [[128, 0], [256, 0], [512, 0], [1024, 0], [2048, 0], [4096, 0], [8192, 0],
         [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]]
_LANES = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]]


def a_dim_kept_whole_beside_cut_rows_has_its_power_of_two_bases():
    """Breaks if the reshape takes the 31 through as it is: reshapeLayout refuses the whole shape."""
    got = compose_reshape((_REGS, _LANES), [16384, 31], [128, 128, 31])
    want = compose_reshape((_REGS, _LANES), [16384, 32], [128, 128, 32])
    assert got == want, (got, want)


def a_dim_not_a_power_of_two_that_the_reshape_merges_is_refused():
    """Breaks if a merged dim is rounded too: 20 x 20 -> 400 moves coordinates, it does not rename them."""
    regs = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]]
    try:
        compose_reshape((regs, []), [20, 20], [400])
    except ValueError:
        return
    raise AssertionError("a 20 x 20 -> 400 reshape was not refused")


CASES = [a_dim_kept_whole_beside_cut_rows_has_its_power_of_two_bases,
         a_dim_not_a_power_of_two_that_the_reshape_merges_is_refused]


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
