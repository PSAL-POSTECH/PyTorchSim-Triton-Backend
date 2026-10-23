"""Time this suite's workloads in PyTorchSim's kernel bundle on --target with TOGSim exactly as validation/gemm does
(its runner, run here with kind, Hq, Hkv and S naming a workload, ranked by score elements); CI times every
workload. results.json holds the machine and per workload the exit code, cycles and what it ran on; exit 1 if any
failed."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from suite_module import load  # noqa: E402
KEYS = ("kind", "Hq", "Hkv", "S")


def score_elements(w):
    """The attention scores a workload computes: causal prefill the lower triangle per query head, decode one row."""
    if w["kind"] == "prefill":
        return w["Hq"] * w["S"] * (w["S"] + 1) // 2
    return w["Hq"] * w["S"]


sim = load("gemm", "simulate_shapes")

if __name__ == "__main__":
    sys.exit(sim.main(here=HERE, keys=KEYS, doc=__doc__, rank=score_elements))
