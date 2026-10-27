"""expect_ir_on: per-machine counts over expect_ir's; the pipeline checks its own; a bad spec is refused.
    python3 tests/passes/expect_ir_on_cases.py        exit 0 iff every case holds"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.driver import machine, compiler  # noqa: E402
from pytorchsim_triton_compiler.contract.kernel_spec import KernelSpec  # noqa: E402

COUNTS = {"09-custom.mlir": {"vcix.iv": 10, "vcix.v.i": 3}, "05-layouts.mlir": {"ttg.dot_op": 0}}
ON = {"tpuv3": {"09-custom.mlir": {"vcix.iv": 5}, "11-kernel.s": {"vpop": 2}}}


def _spec(**kw):
    return KernelSpec(name="probe", kernel=lambda: None, signature={}, constexprs={}, args=[], grid=(1,),
                      reference=lambda: None, make_inputs=lambda: None, **kw)


def a_machine_s_counts_replace_the_keys_they_name_and_add_their_artifacts():
    got = _spec(expect_ir=COUNTS, expect_ir_on=ON).expect_ir_for("tpuv3")
    assert got == {"09-custom.mlir": {"vcix.iv": 5, "vcix.v.i": 3}, "05-layouts.mlir": {"ttg.dot_op": 0},
                   "11-kernel.s": {"vpop": 2}}, got


def another_machine_takes_expect_ir_as_it_is():
    spec = _spec(expect_ir=COUNTS, expect_ir_on=ON)
    assert spec.expect_ir_for("tpuv6e") == COUNTS, spec.expect_ir_for("tpuv6e")
    assert spec.expect_ir == COUNTS and spec.expect_ir_on == ON


def the_pipeline_checks_the_counts_of_the_machine_it_compiles_for():
    spec, was = _spec(expect_ir=COUNTS, expect_ir_on=ON), machine._LAZY.get("TARGET")
    seen = {}
    try:
        for name in ("tpuv3", "tpuv6e"):
            t = copy.copy(machine.subtarget())
            t.name = name
            machine._LAZY["SUBTARGET"] = t
            seen[name] = compiler._expect_ir(spec)["09-custom.mlir"]["vcix.iv"]
    finally:
        machine._LAZY.pop("TARGET", None)
        if was is not None:
            machine._LAZY["SUBTARGET"] = was
    assert seen == {"tpuv3": 5, "tpuv6e": 10}, seen


def expect_ir_on_that_maps_no_machine_to_counts_is_refused():
    for bad in ({"tpuv3": {"09-custom.mlir": 5}}, {"tpuv3": [("09-custom.mlir", {})]}, {3: {}}):
        try:
            _spec(expect_ir_on=bad)
        except ValueError as e:
            assert "expect_ir_on" in str(e), e
            continue
        raise AssertionError(f"taken: {bad}")


CASES = [a_machine_s_counts_replace_the_keys_they_name_and_add_their_artifacts,
         another_machine_takes_expect_ir_as_it_is,
         the_pipeline_checks_the_counts_of_the_machine_it_compiles_for,
         expect_ir_on_that_maps_no_machine_to_counts_is_refused]

if __name__ == "__main__":
    failed = 0
    for case in CASES:
        try:
            case()
            print("PASS ", case.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL ", case.__name__, "--", type(e).__name__, e)
    sys.exit(1 if failed else 0)
