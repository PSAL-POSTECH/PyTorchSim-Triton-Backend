"""The lane count enters once, at the frontend, as the module's "ttg.threads-per-warp" and each
encoding's lane bases; transform checks it against the machine once and every pass reads the IR.
    python3 tests/passes/lane_count_from_ir_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import transform  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import LANES_ATTR as TPW  # noqa: E402
from pytorchsim_triton_compiler.stage import StageError  # noqa: E402

torchsim.register()


def _transform(attrs, body=""):
    """transform.run on a kernel holding `body` whose module says `attrs`: None, or the StageError."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "07-regularized.mlir"), "w") as f:
            f.write("module attributes {%s} {\n  func.func @k() {\n%s    return\n  }\n}\n" % (attrs, body))
        ctx = types.SimpleNamespace(workdir=d, path=lambda n: os.path.join(d, n), lmul=1, loop_lmul={},
                                    spec=types.SimpleNamespace(args=[], grid=(1, 1, 1), extra=()),
                                    log=lambda *_: None)
        try:
            transform.run(ctx)
        except StageError as e:
            return e
    return None


def a_module_made_for_other_lanes_is_refused_at_transform():
    """Breaks if transform takes an IR made for 64 lanes under a 128-lane machine: below transform's
    entry the passes read the lanes from the IR, so nothing after it would notice."""
    assert machine.VECTORLANE_SIZE == 128, machine.VECTORLANE_SIZE
    e = _transform('"%s" = 64 : i32' % TPW)
    assert e is not None and TPW in str(e) and "64" in str(e) and "128" in str(e), e


def a_module_that_does_not_say_its_lanes_is_refused_by_name():
    """Breaks if transform falls back to a default (Triton's 32, or the machine's) for a module with
    no "ttg.threads-per-warp": what the IR was made for is then a guess."""
    e = _transform('"ttg.num-warps" = 1 : i32')
    assert e is not None and TPW in str(e), e


def a_module_made_for_the_machines_lanes_passes_the_check():
    """Breaks if the check refuses the IR every real kernel brings (128 lanes on a 128-lane machine)."""
    e = _transform('"%s" = %d : i32' % (TPW, machine.VECTORLANE_SIZE))
    assert e is None, e


def a_tile_of_other_lanes_than_its_module_says_is_refused_at_transform():
    """Breaks if transform checks only the module's attr: a 64-lane encoding in a module that says
    128 then reaches p02 (which reads encodings) and p00/p04b (which read the attr) disagreeing."""
    l64 = "#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>"
    e = _transform('"%s" = %d : i32' % (TPW, machine.VECTORLANE_SIZE),
                   "    %%e = tensor.empty() : tensor<128xf32, %s>\n" % l64)
    assert e is not None and "64 lanes" in str(e) and TPW in str(e), e


CASES = [a_module_made_for_other_lanes_is_refused_at_transform,
         a_module_that_does_not_say_its_lanes_is_refused_by_name,
         a_module_made_for_the_machines_lanes_passes_the_check,
         a_tile_of_other_lanes_than_its_module_says_is_refused_at_transform]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
