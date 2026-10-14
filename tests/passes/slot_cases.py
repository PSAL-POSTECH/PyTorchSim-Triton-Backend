"""A pipeline Slot is filled by the modules under lowering/ext/ that declare it, kept when the target's isa
has all their REQUIRES; an exactly_one slot refuses none or several, and REQUIRES before the machine is known
is refused. python3 tests/passes/slot_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpu import lower_matmul as tpu_matmul  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6 import lower_matmul as v6_matmul  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower, slots, transform  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.slots import PassGateError, Slot  # noqa: E402


def _machine(name, *isa):
    return types.SimpleNamespace(name=name, isa=["m", "a", "f", "d", "c", "v", "zvl256b", *isa])


def _pass(name, slot=None, requires=()):
    mod = types.SimpleNamespace(__name__=f"ext.{name}", NAME=name, REQUIRES=requires)
    if slot is not None:
        mod.SLOT = slot
    return mod


def _refused(fn):
    try:
        fn()
    except PassGateError as e:
        return str(e)
    raise AssertionError("taken")


def each_accelerator_gets_its_own_matmul_lowering():
    tpu, v6 = _machine("t", "vcix_tpu"), _machine("v", "vcix_tpuv6")
    assert lower.to_custom(tpu)[0] is tpu_matmul, lower.to_custom(tpu)
    assert lower.to_custom(v6)[0] is v6_matmul, lower.to_custom(v6)
    assert lower.to_custom(v6)[1:] == lower.TO_CUSTOM[1:]


def a_pass_whose_requires_the_isa_lacks_is_left_out():
    fold = _pass("tritonpytorchsim-fold", "after", ("vcix_tpuv6", "zvfbfmin"))
    pipeline = [Slot("after")]
    assert slots.resolve(pipeline, _machine("v", "vcix_tpuv6"), [fold]) == []
    assert slots.resolve(pipeline, _machine("v", "vcix_tpuv6", "zvfbfmin"), [fold]) == [fold]


def an_exactly_one_slot_with_two_passes_is_refused_naming_both():
    one, two = _pass("tritonpytorchsim-one", "mm", ("vcix_x",)), _pass("tritonpytorchsim-two", "mm")
    said = _refused(lambda: slots.resolve([Slot("mm", exactly_one=True)], _machine("x", "vcix_x"),
                                          [one, two]))
    assert "tritonpytorchsim-one" in said and "tritonpytorchsim-two" in said, said


def an_exactly_one_slot_with_none_is_refused_naming_the_slot():
    said = _refused(lambda: lower.to_custom(_machine("y", "vcix_tpuv9")))
    assert "lower-matmul" in said and "vcix_tpuv9" in said and "REQUIRES vcix_tpuv6" in said, said


def requires_before_the_machine_is_known_is_refused():
    early = transform.pipelines()[0]
    assert not early[2], early
    said = _refused(lambda: slots.check([early[:2] + (False,)],
                                        [_pass("tritonpytorchsim-x", "s", ("vcix_tpu",))]))
    assert "no pipeline opens" in said, said
    said = _refused(lambda: slots.check([("early", [Slot("s")], False)],
                                        [_pass("tritonpytorchsim-x", "s", ("vcix_tpu",))]))
    assert "before the machine is known" in said, said
    said = _refused(lambda: slots.check([("early", [_pass("tritonpytorchsim-y", None, ("v",))], False)],
                                        []))
    assert "before the machine is known" in said, said
    slots.check([("late", [Slot("s")], True)], [_pass("tritonpytorchsim-x", "s", ("vcix_tpu",))])


def the_shipped_passes_register_against_the_shipped_pipelines():
    found = slots.registered()
    assert tpu_matmul in found and v6_matmul in found, found


def a_module_under_a_package_is_found_without_being_listed_and_runs_in_name_order():
    with tempfile.TemporaryDirectory() as d:
        pkg = os.path.join(d, "probe_ext", "vcix_probe")
        os.makedirs(pkg)
        open(os.path.join(d, "probe_ext", "__init__.py"), "w").close()
        open(os.path.join(pkg, "__init__.py"), "w").close()
        for stem, name in (("aa", "tritonpytorchsim-probe-b"), ("zz", "tritonpytorchsim-probe-a")):
            with open(os.path.join(pkg, f"{stem}.py"), "w") as f:
                f.write(f'NAME = "{name}"\nREQUIRES = ("vcix_probe",)\nSLOT = "s"\n')
        with open(os.path.join(pkg, "helper.py"), "w") as f:
            f.write("X = 1\n")
        sys.path.insert(0, d)
        try:
            found = slots.providers("probe_ext")
        finally:
            sys.path.remove(d)
    names = [m.NAME for m in found]
    assert names == ["tritonpytorchsim-probe-a", "tritonpytorchsim-probe-b"], names
    assert slots.resolve([Slot("s")], _machine("p", "vcix_probe"), found) == found
    assert slots.resolve([Slot("s")], _machine("p", "vcix_tpu"), found) == []


def an_ext_pass_s_forbids_are_found_by_its_name():
    from pytorchsim_triton_compiler.lowering.passes import forbids, pass_name
    patterns, _, source = forbids(pass_name(v6_matmul))
    assert patterns == tuple(v6_matmul.FORBIDS) and patterns and source == v6_matmul.__file__, source


def the_stage_list_and_the_log_name_the_target_s_passes():
    env = dict(os.environ, TORCHSIM_COMPILE_TARGET="tpuv6e_mxu")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"),
                          "--list-stages"], env=env, capture_output=True, text=True).stdout
    assert "tritonpytorchsim-lower-matmul-tpuv6 -> torchsim-lower-to-vcix" in out, out
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"),
                        os.path.join(ROOT, "tests", "cases", "smoke", "gemm.py"), "--workdir", d,
                        "--to", "lower"], env=env, capture_output=True, check=True)
        log = open(os.path.join(d, "stage.log")).read()
    assert "[lower] TO_CUSTOM for tpuv6e_mxu" in log and "tritonpytorchsim-lower-matmul-tpuv6" in log


CASES = [each_accelerator_gets_its_own_matmul_lowering,
         a_pass_whose_requires_the_isa_lacks_is_left_out,
         an_exactly_one_slot_with_two_passes_is_refused_naming_both,
         an_exactly_one_slot_with_none_is_refused_naming_the_slot,
         requires_before_the_machine_is_known_is_refused,
         the_shipped_passes_register_against_the_shipped_pipelines,
         a_module_under_a_package_is_found_without_being_listed_and_runs_in_name_order,
         an_ext_pass_s_forbids_are_found_by_its_name,
         the_stage_list_and_the_log_name_the_target_s_passes]

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
