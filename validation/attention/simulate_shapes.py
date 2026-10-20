"""Time this suite's workloads in PyTorchSim's kernel bundle on --target with TOGSim exactly as validation/gemm does
(its runner, run here with kind, Hq, Hkv and S naming a workload, ranked by score elements); --ci times every
decode and the prefills of at most CI_PREFILL_S queries. results.json holds the machine and per workload the exit code,
cycles and what it ran on; exit 1 if any failed."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("kind", "Hq", "Hkv", "S")
#: The maintainer's CI cap: a prefill's S (Sq = Skv); every decode runs.
CI_PREFILL_S = 4096


def score_elements(w):
    """The attention scores a workload computes: causal prefill the lower triangle per query head, decode one row."""
    if w["kind"] == "prefill":
        return w["Hq"] * w["S"] * (w["S"] + 1) // 2
    return w["Hq"] * w["S"]


def under_cap(w):
    """Whether --ci times a workload: a decode, or a prefill of S <= CI_PREFILL_S."""
    return w["kind"] != "prefill" or w["S"] <= CI_PREFILL_S


def gemm_runner():
    """validation/gemm/simulate_shapes.py as a module of its own name, so it never shadows this file."""
    path = os.path.join(os.path.dirname(HERE), "gemm", "simulate_shapes.py")
    spec = importlib.util.spec_from_file_location("gemm_simulate_shapes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sim = gemm_runner()

if __name__ == "__main__":
    sys.exit(sim.main(here=HERE, keys=KEYS, doc=__doc__, rank=score_elements, in_ci=under_cap))
