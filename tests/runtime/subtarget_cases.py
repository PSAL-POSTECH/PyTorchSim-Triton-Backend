"""The subtarget is one TOGSim machine file: every tool's option is spelled from it, the kernel object
names the file, and a launch runs on that file while it states it, under run_options() of it.
    python3 tests/runtime/subtarget_cases.py     exit 0 iff every case holds"""
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


def _spec():
    return types.SimpleNamespace(name="k", grid=(2,), extra={}, signature={"x": "*fp32"},
                                 args=[types.SimpleNamespace(name="x", role="out", dtype="float32",
                                                             shape=(4,))],
                                 parallel_axes=("x",), graph_prefetch=None)


class _Linked(Exception):
    pass


def _build_argv(t):
    """The llc and gcc argv build.build_elf spells for `t`, the tools recorded instead of run, the ones it
    starts beside another (start_tool) among them."""
    from pytorchsim_triton_compiler.driver import machine
    from pytorchsim_triton_compiler import toolchain
    from pytorchsim_triton_compiler.codegen import build
    ran = []

    def record(ctx, stage, cmd, **kw):
        ran.append(cmd)
        for a in cmd:
            if a.startswith("-pass-remarks-output="):
                open(a.split("=", 1)[1], "w").close()
        if cmd[0] == toolchain.RISCV_GCC:
            raise _Linked()
    saved = (build.run_tool, build.wrapper.find_override, build.wrapper.generate, dict(machine._LAZY),
             build.lane_moves.refuse_lane_moves, build.start_tool, build.finish_tool)
    build.run_tool, build.wrapper.find_override = record, lambda path: None
    build.start_tool = lambda ctx, cmd, **kw: types.SimpleNamespace(wait=lambda: record(ctx, "binary", cmd))
    build.finish_tool = lambda ctx, stage, running: running.wait()
    build.lane_moves.refuse_lane_moves = lambda obj, stage="binary": None
    build.wrapper.generate = lambda spec, subtarget, single_program=False, poisoned=False, **_: ""
    machine._LAZY.clear()
    machine._LAZY["SUBTARGET"] = t
    ctx = types.SimpleNamespace(kernel_path="k.py", spec=None, log=lambda *_: None)
    try:
        with tempfile.TemporaryDirectory() as d:
            f = {k: os.path.join(d, k) for k in ("k.mlir", "k.ll", "k.s", "k.o", "w.c", "k.elf")}
            with open(f["k.mlir"], "w") as m:
                m.write("module {\n}\n")
            open(f["k.ll"], "w").close()
            open(f["k.s"], "w").close()
            try:
                build.build_elf(ctx, "binary", "t", f["k.mlir"], f["k.ll"], f["k.o"], f["k.s"],
                                f["w.c"], f["k.elf"])
            except _Linked:
                pass
    finally:
        build.run_tool, build.wrapper.find_override, build.wrapper.generate, _, \
            build.lane_moves.refuse_lane_moves, build.start_tool, build.finish_tool = saved
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


def _manifest_states_the_machine_once():
    """kernel.json carries the subtarget and no `run`; what a device runs under is run_options() of the
    subtarget as compiled, the machine.yml beside it included."""
    from runtime.device import config
    t = _target.load(V6E)
    with tempfile.TemporaryDirectory() as d:
        ko.write(d, _spec(), t, [0x100, 0x200], {"elf": "k.elf"}, spad_bytes_per_lane=0)
        got = ko.read(d)
        back = _target.as_compiled(got["target"], "<case>")
        run = ko.run_options(back, d, config.COPROCESSOR_BUILD, got)
        yml = open(run["machine_config"]).read()
        machine_config = run["machine_config"].replace(d, "<d>")
    ok = (got["format"] == 10 and "run" not in got and back.file == t.file
          and run["spike_isa"] == "rv64gcv_zfh_zvfbfmin_xcoprocessor"
          and run["memory"] == "0x80000000:0x1900000000"
          and os.path.basename(run["model"]) == "libtpuv6e.so"
          and machine_config == "<d>/machine.yml" and "tpu_mxu_queue_entries: 512" in yml
          and got["target"]["isa"] == t.isa and got["target"]["vpu_num_lanes"] == 128)
    return ok, {"format": got["format"], "keys": sorted(got),
                "run": dict(run, machine_config=machine_config)}


def _override_is_not_as_compiled():
    """An override replaces its answer unasked and is logged: a model named by the environment is
    taken without the vcix build being searched."""
    from runtime.device import as_compiled, base, config
    t = _target.load(V6E)
    said = []
    with tempfile.TemporaryDirectory() as d:
        manifest = {"format": ko.FORMAT, "name": "k", "grid": [1],
                    "target": dict(t.as_dict(), name=t.name, file=t.file), "args": [],
                    "scalars": [], "artifacts": {"machine_config": "machine.yml"}}
        req = base.Launch(d, manifest, d, {}, log=said.append, run=None)
        saved = config.COPROCESSOR_BUILD, config.SPIKE_ISA, config.COPROCESSOR_MODEL
        config.COPROCESSOR_BUILD, config.SPIKE_ISA, config.COPROCESSOR_MODEL = d, "rv64gcv", "/other/lib.so"
        try:
            run = as_compiled.run_options(req)
        finally:
            config.COPROCESSOR_BUILD, config.SPIKE_ISA, config.COPROCESSOR_MODEL = saved
    ok = (run["spike_isa"] == "rv64gcv" and run["model"] == "/other/lib.so"
          and run["memory"] == "0x80000000:0x1900000000"
          and len(said) == 2 and all(line.endswith(": not as compiled") for line in said))
    return ok, {"isa": run["spike_isa"], "model": run["model"], "said": said}


def _isa_states_one_vector_length_and_one_accelerator():
    v6e = _target.load(V6E)
    refused = {}
    for what, isa in (("two accelerators", v6e.isa + ["pytorchsim_tpu"]),
                      ("no accelerator", [e for e in v6e.isa if not e.startswith("pytorchsim_")]),
                      ("no vector length", [e for e in v6e.isa if not e.startswith("zvl")]),
                      ("xsfvcp stated", v6e.isa + ["xsfvcp"])):
        try:
            _target.parse(dict(v6e.as_dict(), isa=isa), "<case>")
            refused[what] = None
        except _target.TargetError as e:
            refused[what] = str(e)
    ok = (all(refused.values()) and "accelerators" in refused["two accelerators"]
          and "vector length" in refused["no vector length"]
          and "every pytorchsim_* is it" in refused["xsfvcp stated"] and v6e.vlen_bits == 256)
    return ok, refused


def _dma_load_latency_is_togsims_key():
    """The cycles a DMA load holds its issue, read under TOGSim's own key, and None where the file
    does not state it. Breaks if the subtarget drops the key or reads it under another name."""
    v6e = _target.load(V6E)
    said = {k: v for k, v in v6e.as_dict().items() if k != "dma_load_latency_cycles"}
    bare = _target.parse(said, "<case>").dma_load_latency_cycles
    got = (v6e.dma_load_latency_cycles, bare, v6e.as_dict().get("dma_load_latency_cycles"))
    return got == (27, None, 27), got


class _Seen(dict):
    """A machine file that records the path of every value parse() reads out of it."""

    def __init__(self, raw, read, at=()):
        super().__init__({k: _Seen(v, read, at + (k,)) if isinstance(v, dict) else v for k, v in raw.items()})
        self.read, self.at = read, at

    def __getitem__(self, k):
        value = super().__getitem__(k)
        if not isinstance(value, _Seen) or k == "vcix":
            self.read.add(self.at + (k,))
        return value


def _paths(said, at=()):
    """The leaf paths of a mapping, vcix as one."""
    return {p for k, v in said.items()
            for p in (_paths(v, at + (k,)) if isinstance(v, dict) and k != "vcix" else {at + (k,)})}


def _as_dict_records_every_key_parse_reads():
    """Breaks if parse() reads a machine-file key that as_dict() -- the kernel object's subtarget record
    and as_compiled's snapshot -- does not record, as vpu_max_lmul and vpu_vector_registers once were."""
    said = dict(_target.load(V6E).as_dict(), target="tpu")
    read = set()
    _target.parse(_Seen(said, read), "<case>")
    unrecorded, unread = sorted(read - _paths(said)), sorted(_paths(said) - read)
    ok = (not unrecorded and not unread and ("vpu_max_lmul",) in read and ("vpu_vector_registers",) in read)
    return ok, {"unrecorded": unrecorded, "unread": unread}


def _max_lmul_change_refuses_old_object():
    """An object compiled while the file said nothing of vpu_max_lmul is refused once it says 4."""
    t = _target.load(V6E)
    said = {k: v for k, v in t.as_dict().items() if k != "vpu_max_lmul"}
    with tempfile.TemporaryDirectory() as d:
        file = os.path.join(d, "m.yml")
        with open(file, "w") as fh:
            json.dump(said, fh)
        old = dict(_target.load(file).as_dict(), name="m", file=file)
        with open(file, "w") as fh:
            json.dump(dict(said, vpu_max_lmul=4), fh)
        try:
            _target.as_compiled(old, "<case>")
            refused = None
        except _target.TargetError as e:
            refused = str(e).replace(d, "<d>")
    ok = old["vpu_max_lmul"] == 8 and refused is not None and "states vpu_max_lmul otherwise" in refused
    return ok, {"refused": refused}


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
    built = os.path.join(config.COPROCESSOR_BUILD, "libtpuv6e.so")
    if not os.path.isfile(built):
        return False, f"no libtpuv6e.so in {config.COPROCESSOR_BUILD}"
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(built, os.path.join(d, "libaccel.so"))
        open(os.path.join(d, "libtpuv6e.so"), "w").close()
        got = os.path.basename(vcix_model.find(d, "pytorchsim_tpuv6e"))
        try:
            vcix_model.find(d, "pytorchsim_tpuv9")
            refused = None
        except vcix_model.ModelError as e:
            refused = str(e).replace(d, "<d>")
    ok = got == "libaccel.so" and refused is not None and refused.endswith("have: tpuv6e")
    return ok, {"tpuv6e": got, "refused": refused}


def _gem5_machine_is_the_models():
    """Each model is timed on its own gem5 machine, examples/<model>/gem5 of the Coprocessor
    checkout: run_options gives tpuv6e's for the tpuv6e subtarget, the tpu model gets tpu's, tpuv3 its own,
    and a model without one (print_args) is refused, naming the path, though another model's machine is
    there."""
    from pytorchsim_triton_compiler.contract import vcix_model
    run = ko.run_options(_target.load(V6E), None, None, model="-", machine_config="-")
    got = {"tpuv6e": os.path.relpath(run["gem5_script"], vcix_model.COPROCESSOR_DIR),
           "tpu": os.path.relpath(vcix_model.gem5_script("pytorchsim_tpu"), vcix_model.COPROCESSOR_DIR),
           "tpuv3": os.path.relpath(vcix_model.gem5_script("pytorchsim_tpuv3"), vcix_model.COPROCESSOR_DIR)}
    try:
        vcix_model.gem5_script("pytorchsim_print_args")
        refused = None
    except vcix_model.MachineError as e:
        refused = str(e).replace(vcix_model.COPROCESSOR_DIR, "<vcix>")
    ok = (got == {"tpuv6e": "examples/tpuv6e/gem5/script_systolic.py",
                  "tpu": "examples/tpu/gem5/script_systolic.py",
                  "tpuv3": "examples/tpuv3/gem5/script_systolic.py"}
          and refused == "no gem5 machine for the vcix model print_args: "
                         "<vcix>/examples/print_args/gem5/script_systolic.py does not exist")
    return ok, {"machines": got, "refused": refused}


def _no_model_is_refused_before_spike():
    from runtime.device import base, config
    from runtime.device.spike import SpikeDevice
    t = _target.load(V6E)
    ran = []
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "k.elf"), "w").close()
        manifest = {"format": ko.FORMAT, "name": "k", "grid": [1],
                    "target": dict(t.as_dict(), name=t.name, file=t.file),
                    "args": [], "scalars": [], "abi": {"parallel_axes": ["x"], "kernel_addr": [0, 4]},
                    "artifacts": {"elf": "k.elf", "machine_config": "machine.yml"}}
        req = base.Launch(d, manifest, os.path.join(d, "runtime"), {}, log=lambda *_: None,
                          run=lambda stage, cmd, **kw: ran.append(cmd))
        saved, config.COPROCESSOR_BUILD, config.COPROCESSOR_MODEL = (config.COPROCESSOR_BUILD, config.COPROCESSOR_MODEL), d, None
        try:
            SpikeDevice().launch(req)
            refused = None
        except base.DeviceError as e:
            refused = str(e).replace(d, "<d>")
        finally:
            config.COPROCESSOR_BUILD, config.COPROCESSOR_MODEL = saved
    ok = (not ran and refused is not None and "no vcix model named tpuv6e" in refused
          and "whose accelerator is pytorchsim_tpuv6e" in refused)
    return ok, {"refused": refused, "ran": [c[:4] for c in ran]}


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
            return {"total_cycles": 0}
    with tempfile.TemporaryDirectory() as d:
        if file is not None:
            file = os.path.join(d, file)
            if stated is not None:
                with open(file, "w") as fh:
                    json.dump(stated, fh)
        manifest = {"format": ko.FORMAT, "name": "k", "grid": [1],
                    "target": dict(t.as_dict(), name=t.name, file=file or t.file),
                    "args": [], "scalars": [], "artifacts": {}}
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


def _retired_togsim_dir_refused():
    """Breaks if the toolchain stops refusing TORCHSIM_COMPILE_TOGSIM_DIR, the retired name of the
    PyTorchSim checkout, or refuses a name that was never one."""
    import subprocess

    def imported(**env):
        return subprocess.run([sys.executable, "-c", "import pytorchsim_triton_compiler.toolchain"],
                              capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, **env))
    retired = imported(TORCHSIM_COMPILE_TOGSIM_DIR="/x")
    other = imported(PYTORCHSIM_COMPILE_TOGSIM_DIR="/x")
    ok = (retired.returncode != 0 and "TORCHSIM_COMPILE_TOGSIM_DIR is retired" in retired.stderr
          and other.returncode == 0)
    return ok, {"retired": retired.stderr[-160:], "other": other.returncode}


def _bare_name():
    got, v3 = _target.load("tpuv6e"), _target.load("tpuv3")
    try:
        _target.load("tpuv9")
        refused = None
    except _target.TargetError as e:
        refused = str(e)
    ok = (got.file == os.path.join(_target.configs_dir(), "tpuv6e.yml") and v3.accel == "pytorchsim_tpuv3"
          and refused is not None and "tpuv3" in refused and "tpuv6e" in refused
          and {"tpuv3", "tpuv6e"} <= set(_target.machines()))
    return ok, {"file": got.file, "tpuv3": v3.accel, "refused": refused}


def _target_subcommand_in_both_drivers():
    """`target <name>` answers the same JSON from the compiler and from tests/pytorchsim-compile, which
    extends cli.parser() and calls cli.run(); a stray word after a kernel file, and `target` alone, are
    refused."""
    import subprocess

    def compiler(*argv):
        return subprocess.run([sys.executable, os.path.join(ROOT, "pytorchsim-triton-compiler"), *argv],
                              capture_output=True, text=True)
    got = {}
    for driver in ("pytorchsim-triton-compiler", "tests/pytorchsim-compile"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, driver), "target", V6E],
                           capture_output=True, text=True)
        got[driver] = (r.returncode, json.loads(r.stdout) if r.returncode == 0 else r.stderr[-200:])
    stray = compiler(os.path.join(ROOT, "tests/cases/smoke/add.py"), V6E)
    alone = compiler("target")
    first, second = got.values()
    ok = (first[0] == second[0] == 0 and first[1] == second[1] and first[1]["name"] == V6E
          and stray.returncode == 2 and f"unrecognized arguments: {V6E}" in stray.stderr
          and alone.returncode == 2 and "target needs a machine" in alone.stderr)
    return ok, {d: rc for d, (rc, _) in got.items()} | {"stray": stray.returncode, "alone": alone.returncode}


def _default_target_is_one_statement():
    import re
    import suite_target
    said = {}
    for rel, rx in (("setup/Dockerfile", r"^ENV PYTORCHSIM_COMPILE_TARGET=(\S+)$"),
                    ("setup/env.sh.in", r"^export PYTORCHSIM_COMPILE_TARGET=(\S+)$")):
        with open(os.path.join(ROOT, rel)) as fh:
            said[rel] = re.findall(rx, fh.read(), re.M)
    for rel in ("tests/run.py", "tests/pytorchsim-compile"):
        with open(os.path.join(ROOT, rel)) as fh:
            said[rel] = re.findall(r'setdefault\("PYTORCHSIM_COMPILE_TARGET", ([^)]+)\)', fh.read())
    ok = (all(v == [suite_target.NAME] for k, v in said.items() if k.startswith("setup/"))
          and all(v == ["suite_target.NAME"] for k, v in said.items() if k.startswith("tests/"))
          and _target.load(suite_target.NAME).name == suite_target.NAME)
    return ok, said


CASES = {"build_takes_isa_from_target": _build_takes_isa,
         "manifest_states_machine_once": _manifest_states_the_machine_once,
         "override_not_as_compiled": _override_is_not_as_compiled,
         "isa_one_vlen_one_accel": _isa_states_one_vector_length_and_one_accelerator,
         "vcix_block_verbatim": _vcix_block_reaches_the_model_as_it_is,
         "model_found_by_descriptor": _model_found_by_its_descriptor,
         "gem5_machine_is_the_models": _gem5_machine_is_the_models,
         "no_model_refused_before_spike": _no_model_is_refused_before_spike,
         "togsim_given_compiled_file": _togsim_given_the_compiled_file,
         "manifest_file_states_its_snapshot": _manifest_file_must_state_its_snapshot,
         "bare_name_is_a_togsim_config": _bare_name,
         "retired_togsim_dir_refused": _retired_togsim_dir_refused,
         "target_subcommand_in_both_drivers": _target_subcommand_in_both_drivers,
         "default_target_is_one_statement": _default_target_is_one_statement,
         "dma_load_latency_is_togsims_key": _dma_load_latency_is_togsims_key,
         "as_dict_records_every_key_parse_reads": _as_dict_records_every_key_parse_reads,
         "max_lmul_change_refuses_old_object": _max_lmul_change_refuses_old_object}


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
