"""A target is its pack: lowering/ext/<family>/PROVIDES names the module filling each Slot. A slot that lowers
ops or is exactly_one is filled by every pack, or the pack is refused when it is loaded, before any stage; a
slot that lowers nothing may stay empty. A provider has what its slot's contract names.
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
from pytorchsim_triton_compiler.lowering.ext import pytorchsim_tpu_family as tpu  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_dma as v6_dma  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_matmul as v6_matmul  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import schedule_mxu_stream as v6_schedule  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower, slots, transform  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.slots import PassGateError, Slot  # noqa: E402


def _machine(name, *isa):
    return types.SimpleNamespace(name=name, isa=["m", "a", "f", "d", "c", "v", "zvl256b", *isa])


def _pass(name, slot=None, forbids=(), **members):
    mod = types.SimpleNamespace(__name__=f"ext.{name}", NAME=name, FORBIDS=forbids, **members)
    if slot is not None:
        mod.SLOT = slot
    return mod


def _opened(*slots_, title="late"):
    return {s.name: (title, s) for s in slots_}


def _pack(provides, name="probe_pack"):
    """A pack package in memory: `provides` as its PROVIDES and the TPU's LAYOUT_RULES, importable
    under lowering/ext/."""
    mod = types.ModuleType(f"{slots.EXT_PACKAGE}.{name}")
    mod.PROVIDES = provides
    mod.LAYOUT_RULES = "tpu"
    sys.modules[mod.__name__] = mod
    return name


def _refused(fn):
    try:
        fn()
    except PassGateError as e:
        return str(e)
    raise AssertionError("taken")


def the_tpu_pack_fills_every_required_slot_and_runs_in_pipeline_order():
    v6 = _machine("v", "pytorchsim_tpuv6e")
    pack = slots.pack(v6)
    required = {n for n, (_, s) in slots.opened().items() if s.required}
    assert required <= set(pack), required - set(pack)
    assert set(tpu.PROVIDES) == set(pack) and pack["lower-dma"] is v6_dma
    assert lower.to_custom(v6)[0] is v6_matmul and lower.to_custom(v6)[1] is v6_schedule, lower.to_custom(v6)
    plain = [i for i in lower.TO_CUSTOM if not isinstance(i, Slot)]
    assert [m for m in lower.to_custom(v6) if not hasattr(m, "SLOT")] == plain
    assert lower.dma_lowering(v6) is v6_dma


def every_accelerator_of_a_family_runs_the_family_s_pack():
    members = [e for e, f in _target.ACCELERATORS.items() if f == "pytorchsim_tpu_family"]
    assert members, _target.ACCELERATORS
    for accel in members:
        assert slots.pack(_machine(accel, accel)) is slots.pack(_machine("v", "pytorchsim_tpuv6e")), accel


def an_isa_naming_no_accelerator_or_an_unknown_one_has_no_pack():
    said = _refused(lambda: slots.pack(_machine("plain")))
    assert "names no accelerator" in said and "plain" in said, said
    said = _refused(lambda: slots.pack(_machine("y", "pytorchsim_tpuv9")))
    assert "pytorchsim_tpuv9" in said and "no family" in said, said


def a_pack_missing_a_required_slot_is_refused_when_loaded_naming_it():
    opened = _opened(Slot("mm", lowers=("x.dot",)), Slot("dma", exactly_one=True), Slot("sched"))
    mm = _pass("pytorchsim-mm", "mm", ("x.dot",))
    said = _refused(lambda: slots.load(_pack({"mm": mm}), opened))
    assert "fills no dma" in said and "mm" not in said.split("fills no")[1], said
    dma = _pass("pytorchsim-dma", "dma")
    got = slots.load(_pack({"mm": mm, "dma": dma}), opened)
    assert got == {"mm": mm, "dma": dma}, got


def an_optional_slot_left_empty_is_left_out_of_the_pipeline():
    opened = _opened(Slot("mm", lowers=("x.dot",)), Slot("sched"))
    mm = _pass("pytorchsim-mm", "mm", ("x.dot",))
    pack = slots.load(_pack({"mm": mm}), opened)
    plain = _pass("pytorchsim-plain")
    slots._PACKS["probe_fam"] = pack
    try:
        _target.ACCELERATORS["pytorchsim_probe"] = "probe_fam"
        got = slots.resolve([plain, Slot("mm"), Slot("sched")], _machine("p", "pytorchsim_probe"))
    finally:
        del _target.ACCELERATORS["pytorchsim_probe"], slots._PACKS["probe_fam"]
    assert got == [plain, mm], got


def a_provider_whose_slot_is_not_the_one_it_is_named_for_is_refused():
    opened = _opened(Slot("mm", lowers=("x.dot",)), Slot("other"))
    wrong = _pass("pytorchsim-other", "other", ("x.dot",))
    said = _refused(lambda: slots.load(_pack({"mm": wrong}), opened))
    assert "pytorchsim-other" in said and "whose SLOT is 'other'" in said, said


def a_provider_that_does_not_forbid_its_slot_s_ops_is_refused():
    opened = _opened(Slot("s", lowers=("x.op", "x.other")))
    said = _refused(lambda: slots.load(_pack({"s": _pass("pytorchsim-x", "s", ("x.op",))}), opened))
    assert "x.other" in said and "FORBID" in said and "x.op" not in said.split("lowers")[1], said


def a_pack_naming_a_slot_no_pipeline_opens_is_refused():
    said = _refused(lambda: slots.load(_pack({"nowhere": _pass("pytorchsim-x", "nowhere")}),
                                       _opened(Slot("s"))))
    assert "nowhere" in said and "no pipeline opens" in said, said


def a_provider_missing_what_its_contract_names_is_refused():
    opened = _opened(Slot("lower-dma", exactly_one=True))
    bare = _pass("pytorchsim-dma", "lower-dma", DESCRIPTOR_DIMS=4, INDEX_KEY=8, insn=lambda *a: "")
    said = _refused(lambda: slots.load(_pack({"lower-dma": bare}), opened))
    assert "sram_footprints" in said and "walk" in said and "DmaProvider" in said, said
    assert "DESCRIPTOR_DIMS" not in said, said
    assert set(slots.members(slots.DmaProvider)) == {"DESCRIPTOR_DIMS", "INDEX_KEY", "insn", "walk",
                                                      "sram_footprints"}
    for name, proto in slots.CONTRACTS.items():
        mod = tpu.PROVIDES[name]
        mod = slots._module(tpu, mod)
        assert all(hasattr(mod, m) for m in slots.members(proto)), (name, slots.members(proto))


def a_pack_s_module_may_be_named_relative_to_it_or_to_another_package():
    opened = _opened(Slot("s"))
    fam = _pack({"s": ".probe_mod"}, "probe_rel")
    probe = types.ModuleType(f"{slots.EXT_PACKAGE}.probe_rel.probe_mod")
    probe.SLOT, probe.NAME, probe.FORBIDS = "s", "pytorchsim-probe", ()
    sys.modules[probe.__name__] = probe
    assert slots.load(fam, opened) == {"s": probe}
    other = _pack({"s": "probe_rel.probe_mod"}, "probe_abs")
    assert slots.load(other, opened) == {"s": probe}


def a_slot_opened_before_the_machine_is_known_or_twice_is_refused():
    early = transform.pipelines()[0]
    assert not early[2], early
    said = _refused(lambda: slots.check([("early", [Slot("s")], False)]))
    assert "before the machine is known" in said, said
    said = _refused(lambda: slots.check([("a", [Slot("s")], True), ("b", [Slot("s")], True)]))
    assert "opened by both a and b" in said, said
    assert set(slots.check([("late", [Slot("s"), _pass("pytorchsim-y")], True)])) == {"s"}
    assert slots.check(transform.pipelines() + lower.pipelines()) == slots.opened()


def the_shipped_providers_are_the_tpu_pack_s():
    found = slots.providers()
    assert v6_matmul in found and v6_dma in found, found
    assert found == sorted(found, key=slots.pass_name)


_NO_PACK = r"""
import copy, json, sys, tempfile
sys.path.insert(0, sys.argv[1])
from pytorchsim_triton_compiler import cli, machine
t = copy.copy(machine.target())
t.isa, t.name = [e for e in t.isa if e != "pytorchsim_tpuv6e"], "no-pack"
machine._LAZY["TARGET"] = t
d = tempfile.mkdtemp()
print("NO-PACK " + json.dumps({"rc": cli.main([sys.argv[2], "--workdir", d, "--to", "transform"])}))
"""


def a_target_without_a_pack_is_refused_before_any_stage_runs():
    out = subprocess.run([sys.executable, "-c", _NO_PACK, ROOT,
                          os.path.join(ROOT, "tests", "cases", "smoke", "add.py")],
                         capture_output=True, text=True)
    said = out.stdout + out.stderr
    assert "names no accelerator" in said and "no-pack" in said, said
    assert "[1/" not in said, said      # no stage started
    got = next((ln for ln in out.stdout.splitlines() if ln.startswith("NO-PACK ")), None)
    assert got is None or json.loads(got[8:])["rc"] != 0, said


def an_ext_pass_s_forbids_are_found_by_its_name():
    from pytorchsim_triton_compiler.lowering.passes import forbids, pass_name
    patterns, _, source = forbids(pass_name(v6_matmul))
    assert patterns == tuple(v6_matmul.FORBIDS) and patterns and source == v6_matmul.__file__, source


def the_stage_list_and_the_log_name_the_target_s_passes():
    env = dict(os.environ, TORCHSIM_COMPILE_TARGET="tpuv6e")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"),
                          "--list-stages"], env=env, capture_output=True, text=True).stdout
    assert ("pytorchsim-lower-matmul -> pytorchsim-schedule-mxu-stream -> "
            "pytorchsim-lower-vector-functions") in out, out
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"),
                        os.path.join(ROOT, "tests", "cases", "smoke", "gemm.py"), "--workdir", d,
                        "--to", "lower"], env=env, capture_output=True, check=True)
        log = open(os.path.join(d, "stage.log")).read()
    assert "[lower] TO_CUSTOM for tpuv6e " in log and "pytorchsim-lower-matmul" in log


CASES = [the_tpu_pack_fills_every_required_slot_and_runs_in_pipeline_order,
         every_accelerator_of_a_family_runs_the_family_s_pack,
         an_isa_naming_no_accelerator_or_an_unknown_one_has_no_pack,
         a_pack_missing_a_required_slot_is_refused_when_loaded_naming_it,
         an_optional_slot_left_empty_is_left_out_of_the_pipeline,
         a_provider_whose_slot_is_not_the_one_it_is_named_for_is_refused,
         a_provider_that_does_not_forbid_its_slot_s_ops_is_refused,
         a_pack_naming_a_slot_no_pipeline_opens_is_refused,
         a_provider_missing_what_its_contract_names_is_refused,
         a_pack_s_module_may_be_named_relative_to_it_or_to_another_package,
         a_slot_opened_before_the_machine_is_known_or_twice_is_refused,
         the_shipped_providers_are_the_tpu_pack_s,
         a_target_without_a_pack_is_refused_before_any_stage_runs,
         an_ext_pass_s_forbids_are_found_by_its_name,
         the_stage_list_and_the_log_name_the_target_s_passes]

if __name__ == "__main__":
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
