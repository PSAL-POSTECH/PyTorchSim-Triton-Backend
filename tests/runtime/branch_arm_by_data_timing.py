"""The arm a run's data takes is the arm TOGSim times: loop_if_compute_arms_by_data_tog run with every
f > 0 (eight exps per program) takes more TOGSim cycles than with every f <= 0 (one add). Needs gem5 and
TOGSim, two --timing runs:  python3 tests/runtime/branch_arm_by_data_timing.py   exit 0 iff it holds"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KERNEL = os.path.join(ROOT, "tests", "cases", "mechanism", "loops", "loop_if_compute_arms_by_data_tog.py")

_SPEC = """import dataclasses
from pytorchsim_triton_compiler.contract.kernel_spec import load_spec
_base = load_spec({kernel!r})


def make_inputs(torch, seed=0):
    got = _base.make_inputs(torch, seed)
    got["f"] = torch.full_like(got["f"], {sign})
    return got


SPEC = dataclasses.replace(_base, make_inputs=make_inputs)
"""


def cycles(sign, root):
    """TOGSim's total cycles of the kernel run with every f set to `sign`."""
    path = os.path.join(root, f"by_data_{'pos' if sign > 0 else 'neg'}.py")
    with open(path, "w") as fh:
        fh.write(_SPEC.format(kernel=KERNEL, sign=float(sign)))
    work = os.path.join(root, os.path.basename(path)[:-3])
    got = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), path, "--workdir",
                          work, "--timing"], cwd=ROOT, capture_output=True, text=True)
    if got.returncode != 0 or not os.path.isfile(os.path.join(work, "togsim.json")):
        raise RuntimeError(f"f = {sign}: the timing run failed (rc {got.returncode}): "
                           f"{(got.stdout + got.stderr).strip()[-400:]}")
    return json.load(open(os.path.join(work, "togsim.json")))["total_cycles"]


def main():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as root:
        pos, neg = cycles(1, root), cycles(-1, root)
    ok = pos > neg
    print(f"{'PASS' if ok else 'FAIL'}  f > 0: {pos} TOGSim cycles, f <= 0: {neg}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
