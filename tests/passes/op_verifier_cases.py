"""What pytorchsim's op verifiers refuse, on IR written here and parsed through the bindings.
    python3 tests/passes/op_verifier_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialect import pytorchsim  # noqa: E402

pytorchsim.register()


def _lane_id(lanes, elem):
    text = (f'func.func @f() -> vector<4x{elem}> {{\n'
            f'  %l = "pytorchsim.lane_id"() <{{lanes = {lanes} : i64}}> : () -> vector<4x{elem}>\n'
            f'  return %l : vector<4x{elem}>\n}}\n')
    with ir.Context(), ir.Location.unknown():
        try:
            ir.Module.parse(text)
        except ir.MLIRError as e:
            return str(e)
    return None


def a_lane_id_whose_element_cannot_hold_the_last_lane_is_refused():
    """Breaks if lane_id takes an element type too narrow for lanes - 1 (its range inference would then
    build an APInt the width cannot hold), or refuses one wide enough."""
    for lanes, elem in ((256, "i8"), (2, "i1"), (128, "index"), (1 << 40, "i64")):
        assert _lane_id(lanes, elem) is None, (lanes, elem)
    for lanes, elem in ((512, "i8"), (3, "i1")):
        said = _lane_id(lanes, elem)
        assert said and f"cannot hold lane {lanes - 1}" in said, (lanes, elem, said)


CASES = [a_lane_id_whose_element_cannot_hold_the_last_lane_is_refused]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
