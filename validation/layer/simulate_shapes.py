"""Time this suite's workloads in PyTorchSim's kernel bundle on --target with TOGSim exactly as validation/gemm does
(its runner, run here with section, name, T and S naming a workload: Qwen3-30B-A3B's decoder layer as one program,
section "layer", and each of its ops alone, section "op"); --ci times only the workloads of at most CI_TOKENS tokens,
the context S of a layer and the tokens T of an op. results.json holds the machine and per workload the exit code,
cycles and what it ran on; exit 1 if any failed."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from suite_module import load  # noqa: E402
KEYS = ("section", "name", "T", "S")
#: The maintainer's CI cap: the tokens a workload holds, a layer's context S and an op's T.
CI_TOKENS = 4096


def tokens(w):
    """The tokens a workload holds: a layer's context S (prefill's T is S, decode's is one token over S), an op's T."""
    return w["S"] if w["section"] == "layer" else w["T"]


def token_pairs(w):
    """What the pool is ordered by: the query-key pairs a layer's attention spans (T*S), an op's tokens T."""
    return w["T"] * w["S"] if w["section"] == "layer" else w["T"]


def under_cap(w):
    """Whether --ci times a workload: at most CI_TOKENS tokens."""
    return tokens(w) <= CI_TOKENS


sim = load("gemm", "simulate_shapes")

if __name__ == "__main__":
    sys.exit(sim.main(here=HERE, keys=KEYS, doc=__doc__, rank=token_pairs, in_ci=under_cap))
