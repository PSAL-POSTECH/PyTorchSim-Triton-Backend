"""simulate_shapes runs every shape on --target (default the suite's tpuv6e, whatever the environment
names) and records the target, accelerator, model and machine file each ran on; a stub stands in for
torchsim-compile.   python3 validation/gemm/simulate_cases.py        exit 0 iff every case holds"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_report  # noqa: E402
import simulate_shapes as sim  # noqa: E402

V6E_YML = "tpuv6e.yml"
_HOLD = tempfile.mkdtemp(prefix="simcases-")
# Another machine file: the v6e's, copied under another name.
ELSEWHERE = os.path.join(_HOLD, "elsewhere.yml")
shutil.copyfile(sim._target.resolve(sim.TARGET), ELSEWHERE)
STUB = '''import json, os, sys, yaml
a = sys.argv; wd = a[a.index("--workdir") + 1]; os.makedirs(wd, exist_ok=True)
if a[a.index("--to") + 1] == "binary":
    path = os.environ["TORCHSIM_COMPILE_TARGET"]
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    said = dict(raw, file=path, name=os.path.splitext(os.path.basename(path))[0])
    json.dump({"target": said}, open(os.path.join(wd, "kernel.json"), "w"))
else:
    json.dump({"total_cycles": 3502, "config": os.environ["STUB_CONFIG"],
               "model": os.environ["STUB_MODEL"]}, open(os.path.join(wd, "togsim.json"), "w"))
'''
SHAPE = {"M": "32", "K": "768", "N": "768", "kernel": "tk_32x768x768_b_w32c256"}


def _run(machine, model=None):
    """One shape through the stub on `machine`; the stub reports `model` (default the planned one)."""
    tmp = tempfile.mkdtemp(prefix="simcases-")
    stub = os.path.join(tmp, "stub.py")
    with open(stub, "w") as fh:
        fh.write(STUB)
    os.makedirs(os.path.join(tmp, "logs"))
    os.environ.update(STUB_CONFIG=machine["togsim_config"], STUB_MODEL=model or machine["model"])
    driver, sim.DRIVER = sim.DRIVER, stub
    try:
        return sim.simulate_shape(SHAPE, os.path.join(tmp, "work"), os.path.join(tmp, "logs"), 60, machine)
    finally:
        sim.DRIVER = driver


def _ran_on(got):
    return (got["rc"], got["target"], got["accel"], os.path.basename(got["model"]),
            os.path.basename(got["togsim_config"]), got.get("us"))


def _default_is_v6e():
    os.environ["TORCHSIM_COMPILE_TARGET"] = ELSEWHERE
    got = _ran_on(_run(sim.plan_machine(sim.TARGET)))
    want = (0, "tpuv6e", "vcix_tpuv6", "libtpuv6.so", V6E_YML, 2.0)
    return sim.TARGET == "tpuv6e" and got == want, got


def _another_machine_file_is_its_own_config():
    """--target of another machine file runs on that file: the file is the TOGSim config."""
    got = _ran_on(_run(sim.plan_machine(ELSEWHERE)))
    want = (0, "elsewhere", "vcix_tpuv6", "libtpuv6.so", "elsewhere.yml", 2.0)
    return got == want, got


def _other_model_fails():
    got = _run(sim.plan_machine(sim.TARGET), model="/elsewhere/libtpu.so")
    ok = got["failed_step"] == "machine" and "model /elsewhere/libtpu.so" in str(got["rc"]) and "us" not in got
    return ok, (got["failed_step"], got["rc"])


def _results_and_report_name_it():
    machine = sim.plan_machine(sim.TARGET)
    got = _run(machine)
    path = os.path.join(tempfile.mkdtemp(prefix="simcases-"), "results.json")

    class Args:
        jobs, timeout = 1, 60
    sim.write_results(path, Args, machine, [got], 1)
    with open(path) as fh:
        res = json.load(fh)
    head = emit_report.ran_on(res["machine"])
    ok = (res["machine"]["accel"] == "vcix_tpuv6" and res["shapes"][0]["model"] == machine["model"]
          and "tpuv6e (vcix_tpuv6)" in head and "libtpuv6.so" in head and V6E_YML in head)
    return ok, head


CASES = {
    "default target is the v6e, not the env's": _default_is_v6e,
    "--target of another machine file times on it": _another_machine_file_is_its_own_config,
    "a shape timed on another model fails": _other_model_fails,
    "results.json and the report name the machine": _results_and_report_name_it,
}


def main():
    os.environ.pop("TORCHSIM_COMPILE_VCIX_MODEL", None)
    sim.config.VCIX_MODEL = None
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:48} {'PASS' if ok else 'FAIL'}  {got}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
