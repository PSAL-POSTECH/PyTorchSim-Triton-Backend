"""The target is the one statement of the machine: every tool's option is spelled from it, the kernel
object carries what the ELF runs under, and a device or a TOGSim config cannot run it as another machine.
    python3 tests/runtime/target_cases.py        exit 0 iff every case holds"""
import json
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract import kernel_object as ko  # noqa: E402
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402

V6E = os.path.join(_target.TARGETS_DIR, "tpuv6e.json")
MXU = os.path.join(_target.TARGETS_DIR, "tpuv6e_mxu.json")


def _llc_mattr():
    t = _target.load(V6E)
    got = (t.llc_mattr(), t.gcc_march(), t.gcc_mabi(), t.llc_march())
    return got == ("+m,+a,+f,+d,+c,+v,+zvfh,+zvfbfmin,+xsfvcp,+zvl512b", "rv64gcv_zfh", "lp64d",
                   "riscv64"), got


def _spike_isa():
    got = _target.load(V6E).spike_isa()
    return got == "rv64gcv_zfh_zvfbfmin_xvcixaccel", got


def _spec():
    return types.SimpleNamespace(name="k", grid=(2,), extra={}, signature={"x": "*fp32"},
                                 args=[types.SimpleNamespace(name="x", role="out", dtype="float32",
                                                             shape=(4,))],
                                 parallel_axes=("x",))


class _Linked(Exception):
    pass


def _build_argv(t):
    """The llc and gcc argv build.build_elf spells for `t`, the tools recorded instead of run."""
    from pytorchsim_triton_compiler import machine, toolchain
    from pytorchsim_triton_compiler.lowering.elf import build
    ran = []

    def record(ctx, stage, cmd, **kw):
        ran.append(cmd)
        if cmd[0] == toolchain.RISCV_GCC:
            raise _Linked()
    saved = (build.run_tool, build.wrapper.find_override, build.wrapper.generate, dict(machine._LAZY))
    build.run_tool, build.wrapper.find_override = record, lambda path: None
    build.wrapper.generate = lambda spec, single_program=False: ""
    machine._LAZY.clear()
    machine._LAZY["TARGET"] = t
    ctx = types.SimpleNamespace(kernel_path="k.py", spec=None, log=lambda *_: None)
    try:
        with tempfile.TemporaryDirectory() as d:
            f = {k: os.path.join(d, k) for k in ("k.ll", "k.s", "k.o", "w.c", "k.elf")}
            open(f["k.ll"], "w").close()
            open(f["k.s"], "w").close()
            try:
                build.build_elf(ctx, "binary", "t", "k.mlir", f["k.ll"], f["k.o"], f["k.s"],
                                f["w.c"], f["k.elf"])
            except _Linked:
                pass
    finally:
        build.run_tool, build.wrapper.find_override, build.wrapper.generate, _ = saved
        machine._LAZY.clear()
        machine._LAZY.update(saved[3])
    llc = [c for c in ran if c[0] == toolchain.LLC]
    gcc = [c for c in ran if c[0] == toolchain.RISCV_GCC]
    return ([a for c in llc for a in c if a.startswith("-mattr=")],
            [a for c in gcc for a in c if a.startswith("-march=")])


def _build_takes_isa():
    v6e = _target.load(V6E)
    raw = dict(v6e.as_dict(), isa=[e for e in v6e.isa if e not in ("zvfbfmin", "zvfh")])
    lean = _target.parse(raw, "<no zvfbfmin, no zvfh>")
    (mattr, march), (lean_mattr, lean_march) = _build_argv(v6e), _build_argv(lean)
    ok = (mattr == [f"-mattr={v6e.llc_mattr()}"] * 2 and march == [f"-march={v6e.gcc_march()}"]
          and lean_mattr == [f"-mattr={lean.llc_mattr()}"] * 2
          and lean_march == ["-march=rv64gcv"] and "zvfbfmin" not in lean_mattr[0])
    return ok, {"tpuv6e": [mattr[:1], march], "lean": [lean_mattr[:1], lean_march]}


def _manifest_run():
    t = _target.load(V6E)
    with tempfile.TemporaryDirectory() as d:
        ko.write(d, _spec(), t, [0x100, 0x200], {"elf": "k.elf"})
        got = ko.read(d)
        yml = open(ko.artifact(d, "machine_config", manifest=got)).read()
    run = got.get("run", {})
    want = {"spike_isa": "rv64gcv_zfh_zvfbfmin_xvcixaccel", "model": "libtpu.so",
            "gem5_script": "script_systolic.py",
            "memory": [[0x80000000, 100 << 30], [0x2000000000, 524288 * 256]]}
    ok = (got["format"] == 3 and run == want and "tpu_systolic_queue_entries: 256" in yml
          and got["target"]["isa"] == t.isa)
    return ok, {"format": got["format"], "run": run, "machine_config": yml.split("\n")[-2]}


def _mxu_model():
    from runtime.device import base, config
    from runtime.device.spike import SpikeDevice
    t = _target.load(MXU)
    run = ko.run_options(t)
    ran = []
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "libtpu.so"), "w").close()
        open(os.path.join(d, "k.elf"), "w").close()
        manifest = {"format": ko.FORMAT, "name": "k", "grid": [1],
                    "target": dict(t.as_dict(), name=t.name), "run": run,
                    "args": [], "scalars": [], "abi": {"parallel_axes": ["x"], "kernel_addr": [0, 4]},
                    "artifacts": {"elf": "k.elf", "machine_config": "machine.yml"}}
        req = base.Launch(d, manifest, os.path.join(d, "runtime"), {}, log=lambda *_: None,
                          run=lambda stage, cmd, **kw: ran.append(cmd))
        saved, config.VCIX_BUILD, config.VCIX_MODEL = (config.VCIX_BUILD, config.VCIX_MODEL), d, None
        try:
            SpikeDevice().launch(req)
            refused = None
        except base.DeviceError as e:
            refused = str(e)
        finally:
            config.VCIX_BUILD, config.VCIX_MODEL = saved
    ok = (run["model"] == "libtpuv6.so" and run["gem5_script"] is None and not ran
          and refused is not None and "libtpuv6.so" in refused)
    return ok, {"model": run["model"], "gem5_script": run["gem5_script"], "refused": refused,
                "ran": [c[:4] for c in ran]}


def _togsim_pairing():
    import yaml
    from runtime.device import config
    v6e, mxu = _target.load(V6E), _target.load(MXU)
    with open(os.path.join(config.TOGSIM_CONFIGS, v6e.togsim_config)) as fh:
        yml = yaml.safe_load(fh)
    wrong, right = mxu.togsim_disagreements(yml), v6e.togsim_disagreements(yml)
    named = " ".join(wrong)
    ok = (not right and "vpu_num_lanes" in named and "vpu_spad_size_kb_per_lane" in named
          and mxu.togsim_config is None)
    return ok, {"tpuv6e_mxu": wrong, "tpuv6e": right}


def _togsim_absent_unit_is_systolic():
    mxu, v6e = _target.load(MXU), _target.load(V6E)
    def keyless(t):
        return {"vpu_num_lanes": t.vector_lanes, "vpu_spad_size_kb_per_lane": t.spad_size // 1024}
    wrong, right = mxu.togsim_disagreements(keyless(mxu)), v6e.togsim_disagreements(keyless(v6e))
    drafted = _target.from_togsim(dict(keyless(mxu), vpu_vector_length_bits=256,
                                       tpu_systolic_queue_entries=256), v6e)
    ok = (len(wrong) == 1 and "systolic_array_unit" in wrong[0] and not right
          and drafted.get("matmul_unit", _target.DEFAULT_MATMUL_UNIT) == "systolic")
    return ok, {"tpuv6e_mxu": wrong, "tpuv6e": right, "drafted": drafted.get("matmul_unit")}


def _togsim_refused_before_gem5():
    from runtime import stages
    from runtime.device import REGISTRY, base, config
    t = _target.load(V6E)
    measured = []

    class Gem5:
        def measure(self, req, ports=False):
            measured.append(req)
            return [1], [{}]
    manifest = {"format": ko.FORMAT, "name": "k", "grid": [1], "target": dict(t.as_dict(), name=t.name),
                "run": ko.run_options(t), "args": [], "scalars": [], "artifacts": {}}
    with tempfile.TemporaryDirectory() as d:
        yml = os.path.join(d, "other.yml")
        with open(yml, "w") as fh:
            fh.write(f"vpu_num_lanes: {t.vector_lanes // 2}\nvpu_spad_size_kb_per_lane: {t.spad_size // 1024}\n")
        req = base.Launch(d, manifest, d, {}, log=lambda *_: None, run=None)
        saved = REGISTRY["gem5"], config.TOGSIM_CONFIG
        REGISTRY["gem5"], config.TOGSIM_CONFIG = Gem5(), yml
        try:
            stages.time_launch(req)
            refused = None
        except base.DeviceError as e:
            refused = str(e)
        finally:
            REGISTRY["gem5"], config.TOGSIM_CONFIG = saved
    ok = refused is not None and "vpu_num_lanes" in refused and not measured
    return ok, {"refused": refused, "gem5_ran": len(measured)}


def _from_togsim_unit():
    like = _target.load(V6E)
    yml = {"vpu_num_lanes": 128, "vpu_vector_length_bits": 256, "vpu_spad_size_kb_per_lane": 1024,
           "core_spad_size_kb": 131072, "systolic_array_unit": "Mxu", "tpu_mxu_queue_entries": 512,
           "tpu_systolic_queue_entries": 256}
    draft = _target.from_togsim(yml, like)
    try:
        _target.from_togsim(dict(yml, systolic_array_unit="TPUV6_MXU"), like)
        refused = None
    except _target.TargetError as e:
        refused = str(e)
    ok = (draft.get("matmul_unit") == "mxu" and draft["systolic_queue_entries"] == 512
          and refused is not None and "TPUV6_MXU" in refused
          and _target.parse(dict(draft, name="draft"), "<draft>").matmul_unit == "mxu")
    return ok, {"matmul_unit": draft.get("matmul_unit"), "queue": draft["systolic_queue_entries"],
                "refused": refused}


def _bare_name():
    got = _target.load("tpuv6e_mxu")
    try:
        _target.load("tpuv9")
        refused = None
    except _target.TargetError as e:
        refused = str(e)
    ok = got.name == "tpuv6e_mxu" and refused is not None and "tpuv6e" in refused
    return ok, {"name": got.name, "refused": refused}


def _targets_are_package_data():
    import fnmatch
    import tomllib
    pkg = os.path.join(ROOT, "pytorchsim_triton_compiler")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        globs = tomllib.load(fh).get("tool", {}).get("setuptools", {}).get("package-data", {}) \
            .get("pytorchsim_triton_compiler", [])
    shipped = sorted(os.path.relpath(os.path.join(_target.TARGETS_DIR, f), pkg)
                     for f in os.listdir(_target.TARGETS_DIR) if f.endswith(".json"))
    left = [f for f in shipped if f.startswith("..") or not any(fnmatch.fnmatch(f, g) for g in globs)]
    return bool(shipped) and not left, {"shipped": shipped, "not_package_data": left}


def _default_target_is_one_statement():
    import re
    import suite_target
    said = {}
    for rel, rx in (("setup/Dockerfile", r"^ENV TORCHSIM_COMPILE_TARGET=(\S+)$"),
                    ("setup/env.sh.in", r"^export TORCHSIM_COMPILE_TARGET=(\S+)$")):
        with open(os.path.join(ROOT, rel)) as fh:
            said[rel] = re.findall(rx, fh.read(), re.M)
    for rel in ("tests/run.py", "tests/torchsim-compile"):
        with open(os.path.join(ROOT, rel)) as fh:
            said[rel] = re.findall(r'setdefault\("TORCHSIM_COMPILE_TARGET", ([^)]+)\)', fh.read())
    ok = (all(v == [suite_target.NAME] for k, v in said.items() if k.startswith("setup/"))
          and all(v == ["suite_target.NAME"] for k, v in said.items() if k.startswith("tests/"))
          and _target.load(suite_target.NAME).name == suite_target.NAME)
    return ok, said


CASES = {"llc_mattr_matches_compiled": _llc_mattr, "spike_isa_covers_bf16": _spike_isa,
         "build_takes_isa_from_target": _build_takes_isa,
         "manifest_carries_run": _manifest_run, "mxu_selects_its_model": _mxu_model,
         "togsim_pairing": _togsim_pairing,
         "togsim_absent_unit_is_systolic": _togsim_absent_unit_is_systolic, "from_togsim_carries_unit": _from_togsim_unit,
         "togsim_refused_before_gem5": _togsim_refused_before_gem5,
         "bare_name_resolves_under_targets": _bare_name,
         "targets_are_package_data": _targets_are_package_data,
         "default_target_is_one_statement": _default_target_is_one_statement}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:34} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
