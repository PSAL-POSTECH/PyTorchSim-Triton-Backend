"""A Slot is filled by the modules under lowering/ext/ whose REQUIRES the isa has; one that lowers ops
stays empty without one and refuses, in its place, a module holding them; exactly_one refuses none.
python3 tests/passes/slot_cases.py        exit 0 iff every case holds"""
import json
import os
import subprocess
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_matmul as v6_matmul  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import schedule_mxu_stream as v6_schedule  # noqa: E402
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
    tpu, v6 = _machine("t", "vcix_tpu"), _machine("v", "vcix_tpuv6e")
    a, b = _pass("tritonpytorchsim-mm-tpu", "mm", ("vcix_tpu",)), _pass("tritonpytorchsim-mm-v6", "mm",
                                                                       ("vcix_tpuv6e",))
    mm = [Slot("mm", exactly_one=True)]
    assert slots.resolve(mm, tpu, [a, b]) == [a] and slots.resolve(mm, v6, [a, b]) == [b]
    assert lower.to_custom(v6)[0] is v6_matmul, lower.to_custom(v6)
    assert lower.to_custom(v6)[1] is v6_schedule, lower.to_custom(v6)
    plain = [i for i in lower.TO_CUSTOM if not isinstance(i, Slot)]
    assert [m for m in lower.to_custom(v6) if not hasattr(m, "SLOT")] == plain


def a_pass_whose_requires_the_isa_lacks_is_left_out():
    fold = _pass("tritonpytorchsim-fold", "after", ("vcix_tpuv6e", "zvfbfmin"))
    pipeline = [Slot("after")]
    assert slots.resolve(pipeline, _machine("v", "vcix_tpuv6e"), [fold]) == []
    assert slots.resolve(pipeline, _machine("v", "vcix_tpuv6e", "zvfbfmin"), [fold]) == [fold]


def an_exactly_one_slot_with_two_passes_is_refused_naming_both():
    one, two = _pass("tritonpytorchsim-one", "mm", ("vcix_x",)), _pass("tritonpytorchsim-two", "mm")
    said = _refused(lambda: slots.resolve([Slot("mm", exactly_one=True)], _machine("x", "vcix_x"),
                                          [one, two]))
    assert "tritonpytorchsim-one" in said and "tritonpytorchsim-two" in said, said


def an_exactly_one_slot_with_none_is_refused_naming_the_slot():
    said = _refused(lambda: lower.dma_lowering(_machine("y", "vcix_tpuv9")))
    assert "lower-dma" in said and "vcix_tpuv9" in said and "REQUIRES vcix_tpuv6e" in said, said


def a_lowering_slot_with_no_pass_is_left_empty_and_takes_at_most_one():
    no_model = _machine("y", "vcix_tpuv9")
    got = lower.to_custom(no_model)
    empty = [m for m in got if type(m).__name__ == "Unfilled"]
    assert [m.slot.name for m in empty] == ["lower-matmul", "lower-vector-functions",
                                            "lower-cross-lane", "lower-kernel-vcix"], got
    assert [m for m in got if m not in empty] == [lower.p08_refuse_unlowered_math,
                                                  lower.p11_fit_transfers_to_descriptor], got
    copies = transform.phases(no_model)[-1][1]
    assert any(getattr(m, "slot", None) is transform.LOWER_COPIES for m in copies), copies
    one, two = _pass("tritonpytorchsim-one", "mm"), _pass("tritonpytorchsim-two", "mm")
    one.FORBIDS = two.FORBIDS = ("x.op",)
    said = _refused(lambda: slots.resolve([Slot("mm", lowers=("x.op",))], no_model, [one, two]))
    assert "tritonpytorchsim-one" in said and "tritonpytorchsim-two" in said, said


def a_provider_that_does_not_forbid_its_slot_s_ops_is_refused():
    said = _refused(lambda: slots.check([("late", [Slot("s", lowers=("x.op", "x.other"))], True)],
                                        [_pass("tritonpytorchsim-x", "s")]))
    assert "x.op" in said and "FORBID" in said, said


_NO_MODEL = r"""
import copy, json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
from pytorchsim_triton_compiler import cli, machine
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_dma
from pytorchsim_triton_compiler.lowering.passes import lower
t = copy.copy(machine.target())
t.isa, t.name = [e for e in t.isa if e != "vcix_tpuv6e"], "no-model"
machine._LAZY["TARGET"] = t
d = tempfile.mkdtemp()
got = {"transform": cli.main([sys.argv[2], "--workdir", d, "--to", "transform"])}
got["log"] = open(os.path.join(d, "stage.log")).read()
if got["transform"] == 0:
    try:
        lower.run_module_passes(os.path.join(d, "08-transformed.mlir"),
                                os.path.join(d, "09-custom.mlir"), lower.to_custom(t), target=t,
                                descriptor_dims=lower_dma.DESCRIPTOR_DIMS)
        got["custom"] = sorted(lower.ops_in_file(os.path.join(d, "09-custom.mlir")))
    except Exception as e:
        got["refused"] = str(e)
print("NO-MODEL " + json.dumps(got))
"""


def _without_the_model(kernel):
    """What the target's own machine with vcix_tpuv6e taken out of its isa makes of `kernel`: the
    transform step's return code, log and what it printed, then TO_CUSTOM's ops or its refusal."""
    out = subprocess.run([sys.executable, "-c", _NO_MODEL, ROOT,
                          os.path.join(ROOT, "tests", "cases", *kernel.split("/"))],
                         capture_output=True, text=True).stdout
    got = json.loads(next(ln for ln in out.splitlines() if ln.startswith("NO-MODEL "))[9:])
    got["printed"] = out
    return got


def an_isa_without_the_model_compiles_a_kernel_holding_none_of_its_slots_ops():
    got = _without_the_model("smoke/add.py")
    assert got["transform"] == 0 and "[lower-copies: no pass]" in got["log"], got
    assert "refused" not in got and "torchsim.transfer" in got["custom"], got


def an_isa_without_the_model_refuses_a_dot_naming_lower_matmul():
    got = _without_the_model("smoke/gemm.py")
    assert got["transform"] == 0, got
    said = got.get("refused", "")
    assert "lower-matmul" in said and "torchsim.dot" in said and "no-model" in said, got


def an_isa_without_the_model_refuses_a_copy_naming_lower_copies():
    got = _without_the_model("mechanism/ops/ops_xlu_transpose_i64.py")
    assert got["transform"] != 0, got
    said = got["printed"]
    assert "slot lower-copies" in said and "torchsim.copy" in said and "no-model" in said, said


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
    assert v6_matmul in found, found


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
    env = dict(os.environ, TORCHSIM_COMPILE_TARGET="tpuv6e")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"),
                          "--list-stages"], env=env, capture_output=True, text=True).stdout
    assert ("tritonpytorchsim-lower-matmul-tpuv6e -> tritonpytorchsim-schedule-mxu-stream -> "
            "tritonpytorchsim-lower-vector-functions-tpuv6e") in out, out
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"),
                        os.path.join(ROOT, "tests", "cases", "smoke", "gemm.py"), "--workdir", d,
                        "--to", "lower"], env=env, capture_output=True, check=True)
        log = open(os.path.join(d, "stage.log")).read()
    assert "[lower] TO_CUSTOM for tpuv6e " in log and "tritonpytorchsim-lower-matmul-tpuv6e" in log


CASES = [each_accelerator_gets_its_own_matmul_lowering,
         a_pass_whose_requires_the_isa_lacks_is_left_out,
         an_exactly_one_slot_with_two_passes_is_refused_naming_both,
         an_exactly_one_slot_with_none_is_refused_naming_the_slot,
         a_lowering_slot_with_no_pass_is_left_empty_and_takes_at_most_one,
         a_provider_that_does_not_forbid_its_slot_s_ops_is_refused,
         an_isa_without_the_model_compiles_a_kernel_holding_none_of_its_slots_ops,
         an_isa_without_the_model_refuses_a_dot_naming_lower_matmul,
         an_isa_without_the_model_refuses_a_copy_naming_lower_copies,
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
