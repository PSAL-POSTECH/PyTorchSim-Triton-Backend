"""Time this suite's workloads in PyTorchSim's kernel bundle on --target with TOGSim exactly as validation/layer does
(its runner, cap and rank, run here with section, name, T and S naming a workload: Qwen3-8B's decoder layer as one
program, section "layer", and each of its ops alone, section "op"); --ci times only the workloads of at most
CI_TOKENS tokens, --where section=op only its ops. results.json holds the machine and per workload the exit code,
cycles and what it ran on; exit 1 if any failed."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from suite_module import load  # noqa: E402


layer = load("layer", "simulate_shapes")
KEYS, CI_TOKENS, sim = layer.KEYS, layer.CI_TOKENS, layer.sim
tokens, token_pairs, under_cap = layer.tokens, layer.token_pairs, layer.under_cap

if __name__ == "__main__":
    sys.exit(sim.main(here=HERE, keys=KEYS, doc=__doc__, rank=token_pairs, in_ci=under_cap))
