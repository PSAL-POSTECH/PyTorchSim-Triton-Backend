"""simulate_shapes runs every shape on --target (default the v6e MXU, whatever the environment names)
and records the target, model and TOGSim config each ran on; a stub stands in for torchsim-compile.
    python3 validation/gemm/simulate_cases.py        exit 0 iff every case holds"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_report  # noqa: E402
import simulate_shapes as sim  # noqa: E402

TARGETS = os.path.join(sim.REPO, "tests", "targets")
OVERRIDES = ("TORCHSIM_COMPILE_VCIX_MODEL", "TORCHSIM_COMPILE_TOGSIM_CONFIG")
STUB = '''import json, os, sys
a = sys.argv; wd = a[a.index("--workdir") + 1]; os.makedirs(wd, exist_ok=True)
if a[a.index("--to") + 1] == "binary":
    with open(os.environ["TORCHSIM_COMPILE_TARGET"]) as fh:
        json.dump({"target": json.load(fh)}, open(os.path.join(wd, "kernel.json"), "w"))
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
    return (got["rc"], got["target"], got["matmul_unit"], os.path.basename(got["model"]),
            os.path.basename(got["togsim_config"]), got.get("us"))


def _default_is_mxu():
    os.environ["TORCHSIM_COMPILE_TARGET"] = os.path.join(TARGETS, "tpuv6e.json")
    got = _ran_on(_run(sim.plan_machine(sim.TARGET)))
    want = (0, "tpuv6e_mxu", "mxu", "libtpu_mxu.so", "systolic_ws_256x256_c1_simple_noc_tpuv6e_mxu.yml", 2.0)
    return got == want, got


def _systolic_selectable():
    got = _ran_on(_run(sim.plan_machine(os.path.join(TARGETS, "tpuv6e.json"))))
    want = (0, "tpuv6e", "systolic", "libtpu.so", "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml", 2.0)
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
    ok = (res["machine"]["matmul_unit"] == "mxu" and res["shapes"][0]["model"] == machine["model"]
          and "tpuv6e_mxu (mxu)" in head and "libtpu_mxu.so" in head and "tpuv6e_mxu.yml" in head)
    return ok, head


CASES = {
    "default target is the MXU, not the env's": _default_is_mxu,
    "--target tpuv6e.json runs the systolic stand-in": _systolic_selectable,
    "a shape timed on another model fails": _other_model_fails,
    "results.json and the report name the machine": _results_and_report_name_it,
}


def main():
    for k in OVERRIDES:
        os.environ.pop(k, None)
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
