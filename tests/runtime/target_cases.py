"""The target is one TOGSim machine file: every tool's option is spelled from it, the kernel object
carries what the ELF runs under and names the file, and a launch runs on that file while it states it.
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

V6E = "tpuv6e"


def _llc_mattr():
    t = _target.load(V6E)
    got = (t.llc_mattr(), t.gcc_march(), t.gcc_mabi(), t.llc_march())
    return got == ("+m,+a,+f,+d,+c,+v,+zvfh,+zvfbfmin,+zvl256b,+xsfvcp", "rv64gcv_zfh", "lp64d",
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
    build.wrapper.generate = lambda spec, target, single_program=False: ""
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
    want = {"spike_isa": "rv64gcv_zfh_zvfbfmin_xvcixaccel", "accel": "vcix_tpuv6",
            "memory": [[0x80000000, 100 << 30], [0x2000000000, 1048576 * 128]]}
    ok = (got["format"] == 5 and run == want and "tpu_mxu_queue_entries: 512" in yml
          and got["target"]["isa"] == t.isa and got["target"]["file"] == t.file
          and got["target"]["vpu_num_lanes"] == 128)
    return ok, {"format": got["format"], "run": run, "file": got["target"]["file"],
                "machine_config": yml.split("\n")[-2]}


def _isa_states_one_vector_length_and_one_accelerator():
    v6e = _target.load(V6E)
    refused = {}
    for what, isa in (("two accelerators", v6e.isa + ["vcix_tpu"]),
                      ("no accelerator", [e for e in v6e.isa if not e.startswith("vcix_")]),
                      ("no vector length", [e for e in v6e.isa if not e.startswith("zvl")]),
                      ("xsfvcp stated", v6e.isa + ["xsfvcp"])):
        try:
            _target.parse(dict(v6e.as_dict(), isa=isa), "<case>")
            refused[what] = None
        except _target.TargetError as e:
            refused[what] = str(e)
    ok = (all(refused.values()) and "accelerators" in refused["two accelerators"]
          and "vector length" in refused["no vector length"]
          and "every vcix_* is it" in refused["xsfvcp stated"] and v6e.vlen_bits == 256)
    return ok, refused


def _vcix_block_reaches_the_model_as_it_is():
    v6e = _target.load(V6E)
    t = _target.parse(dict(v6e.as_dict(), vcix={"tpu_x_y": 7, "tpu_trace": 1}), "<case>")
    with tempfile.TemporaryDirectory() as d:
        yml = open(t.write_model_yaml(os.path.join(d, "m.yml"))).read()
    ok = yml.endswith("tpu_x_y: 7\ntpu_trace: 1\n") and "vpu_vector_length_bits: 256" in yml
    return ok, yml.splitlines()


def _model_found_by_its_descriptor():
    import shutil
    from runtime.device import config
    from pytorchsim_triton_compiler.contract import vcix_model
    built = os.path.join(config.VCIX_BUILD, "libtpuv6.so")
    if not os.path.isfile(built):
        return False, f"no libtpuv6.so in {config.VCIX_BUILD}"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(built, os.path.join(d, "libaccel.so"))
        open(os.path.join(d, "libtpuv6.so"), "w").close()
        got = os.path.basename(vcix_model.find(d, "vcix_tpuv6"))
        try:
            vcix_model.find(d, "vcix_tpuv9")
            refused = None
        except vcix_model.ModelError as e:
            refused = str(e).replace(d, "<d>")
    ok = got == "libaccel.so" and refused is not None and refused.endswith("have: tpuv6")
    return ok, {"tpuv6": got, "refused": refused}


def _no_model_is_refused_before_spike():
    from runtime.device import base, config
    from runtime.device.spike import SpikeDevice
    t = _target.load(V6E)
    run = ko.run_options(t)
    ran = []
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "k.elf"), "w").close()
        manifest = {"format": ko.FORMAT, "name": "k", "grid": [1],
                    "target": dict(t.as_dict(), name=t.name, file=t.file), "run": run,
                    "args": [], "scalars": [], "abi": {"parallel_axes": ["x"], "kernel_addr": [0, 4]},
                    "artifacts": {"elf": "k.elf", "machine_config": "machine.yml"}}
        req = base.Launch(d, manifest, os.path.join(d, "runtime"), {}, log=lambda *_: None,
                          run=lambda stage, cmd, **kw: ran.append(cmd))
        saved, config.VCIX_BUILD, config.VCIX_MODEL = (config.VCIX_BUILD, config.VCIX_MODEL), d, None
        try:
            SpikeDevice().launch(req)
            refused = None
        except base.DeviceError as e:
            refused = str(e).replace(d, "<d>")
        finally:
            config.VCIX_BUILD, config.VCIX_MODEL = saved
    ok = (run["accel"] == "vcix_tpuv6" and not ran and refused is not None
          and "no vcix model named tpuv6" in refused)
    return ok, {"accel": run["accel"], "refused": refused, "ran": [c[:4] for c in ran]}


def _unit_agrees_within_the_file():
    v6e = _target.load(V6E)
    said = v6e.as_dict()
    unsaid = {k: v for k, v in said.items() if k != "systolic_array_unit"}
    refused = {}
    for what, raw in (("SYSTOLIC", dict(said, systolic_array_unit="SYSTOLIC")), ("unit unsaid", unsaid)):
        try:
            _target.parse(raw, "<case>")
            refused[what] = None
        except _target.TargetError as e:
            refused[what] = str(e)
    ok = (all(refused.values()) and "TPUV6_MXU" in refused["SYSTOLIC"]
          and "(unsaid)" in refused["unit unsaid"] and said["systolic_array_unit"] == "TPUV6_MXU"
          and "vpu_vector_length_bits" not in said and v6e.vlen_bits == 256)
    return ok, refused


def _timed_on(file=None, stated=None):
    """time_launch with gem5 and TOGSim recorded instead of run, on a manifest naming `file` (tpuv6e's
    by default) whose snapshot is tpuv6e's; `stated` is what `file` is written to state: (the file
    TOGSim got, gem5 ran, the refusal)."""
    from runtime import stages
    from runtime.device import REGISTRY, base
    t = _target.load(V6E)
    ran = {"gem5": 0, "machine": None}

    class Gem5:
        def measure(self, req, ports=False):
            ran["gem5"] += 1
            return [1], [{}]

    class TOGSim:
        def simulate(self, req, machine, cycles, windows):
            ran["machine"] = machine
            return {}
    with tempfile.TemporaryDirectory() as d:
        if file is not None:
            file = os.path.join(d, file)
            if stated is not None:
                with open(file, "w") as fh:
                    json.dump(stated, fh)
        manifest = {"format": ko.FORMAT, "name": "k", "grid": [1],
                    "target": dict(t.as_dict(), name=t.name, file=file or t.file),
                    "run": ko.run_options(t), "args": [], "scalars": [], "artifacts": {}}
        req = base.Launch(d, manifest, d, {}, log=lambda *_: None, run=None)
        saved = REGISTRY["gem5"], REGISTRY["togsim"]
        REGISTRY["gem5"], REGISTRY["togsim"] = Gem5(), TOGSim()
        try:
            stages.time_launch(req)
            refused = None
        except base.DeviceError as e:
            refused = str(e).replace(d, "<d>")
        finally:
            REGISTRY["gem5"], REGISTRY["togsim"] = saved
    return ran["machine"] and ran["machine"].replace(d, "<d>"), ran["gem5"], refused, t.file


def _togsim_given_the_compiled_file():
    machine, gem5, refused, file = _timed_on()
    ok = machine == file and gem5 == 1 and refused is None and file.endswith("/configs/tpuv6e.yml")
    return ok, {"togsim --config": machine, "manifest file": file, "refused": refused}


def _manifest_file_must_state_its_snapshot():
    """The file the manifest names is what TOGSim times on, and only while it states what the
    manifest recorded of it: a changed key, or no file, is refused before gem5 runs."""
    said = _target.load(V6E).as_dict()
    took, gem5_same, _, _ = _timed_on("same.yml", said)
    got = {}
    for what, stated in (("queue 256", dict(said, vcix={"tpu_mxu_queue_entries": 256})),
                         ("lanes 64", dict(said, vpu_num_lanes=64)), ("no file", None)):
        machine, gem5, refused, _ = _timed_on("m.yml", stated)
        got[what] = (machine, gem5, refused)
    ok = (took == "<d>/same.yml" and gem5_same == 1
          and all(m is None and not g and r for m, g, r in got.values())
          and "states vcix otherwise" in got["queue 256"][2]
          and "states vpu_num_lanes otherwise" in got["lanes 64"][2]
          and "No such file" in got["no file"][2])
    return ok, {"same": took, **{k: v[2] for k, v in got.items()}}


def _bare_name():
    got = _target.load("tpuv6e")
    refused = {}
    for name in ("tpuv9", "tpuv3"):
        try:
            _target.load(name)
            refused[name] = None
        except _target.TargetError as e:
            refused[name] = str(e)
    ok = (got.file == os.path.join(_target.configs_dir(), "tpuv6e.yml")
          and all(refused.values()) and "have: tpuv6e." in refused["tpuv9"]
          and "vcix_tpuv3" in refused["tpuv3"] and "tpuv3" not in _target.machines())
    return ok, {"file": got.file, "refused": refused}


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
         "manifest_carries_run": _manifest_run,
         "isa_one_vlen_one_accel": _isa_states_one_vector_length_and_one_accelerator,
         "vcix_block_verbatim": _vcix_block_reaches_the_model_as_it_is,
         "model_found_by_descriptor": _model_found_by_its_descriptor,
         "no_model_refused_before_spike": _no_model_is_refused_before_spike,
         "unit_agrees_in_file": _unit_agrees_within_the_file,
         "togsim_given_compiled_file": _togsim_given_the_compiled_file,
         "manifest_file_states_its_snapshot": _manifest_file_must_state_its_snapshot,
         "bare_name_is_a_togsim_config": _bare_name,
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
