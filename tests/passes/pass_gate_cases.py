"""run_module_passes runs a pass iff one of its MARKERS names an op of the module -- not a string,
symbol, location or longer op name that spells it -- even one printed short (`return`).
    python3 tests/passes/pass_gate_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.passes import lower  # noqa: E402


def _ran(marker, body):
    """Whether a pass whose one marker is `marker` ran on a module holding `body`."""
    seen = []
    p = types.SimpleNamespace(__name__="c99_gate_probe", MARKERS=(marker,),
                              run=lambda module, **_: seen.append(module))
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.mlir"), os.path.join(d, "out.mlir")
        with open(src, "w") as f:
            f.write("module {\n" + body + "\n}\n")
        lower.run_module_passes(src, dst, [p])
        assert os.path.exists(dst)
    return bool(seen)


def a_string_attribute_is_not_the_op():
    assert not _ran("math.exp", '"test.op"() {note = "math.exp"} : () -> ()')


def a_symbol_name_is_not_the_op():
    assert not _ran("pytorchsim.dot", "func.func private @pytorchsim.dot()")


def a_location_is_not_the_op():
    assert not _ran("pytorchsim.dot", '"test.op"() : () -> () loc("pytorchsim.dot")')


def a_longer_op_name_is_not_the_op():
    assert not _ran("math.exp", """func.func @f(%x: f32) -> f32 {
  %y = math.expm1 %x : f32
  return %y : f32
}""")


def an_op_printed_short_is_the_op():
    assert _ran("func.return", "func.func @f() {\n  return\n}")


def the_op_runs_the_pass():
    assert _ran("math.exp", """func.func @f(%x: f32) -> f32 {
  %y = math.exp %x : f32
  return %y : f32
}""")


CASES = [a_string_attribute_is_not_the_op, a_symbol_name_is_not_the_op, a_location_is_not_the_op,
         a_longer_op_name_is_not_the_op, an_op_printed_short_is_the_op, the_op_runs_the_pass]


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
