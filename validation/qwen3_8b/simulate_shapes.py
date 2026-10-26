"""Time this suite's workloads in PyTorchSim's kernel bundle on --target with TOGSim exactly as validation/layer does
(its runner, cap and rank, run here with section, name, T and S naming a workload: Qwen3-8B's decoder layer as one
program, section "layer", and each of its ops alone, section "op"); --ci times only the layers of at most CI_TOKENS
tokens, --where section=op only its ops. results.json holds the machine and per workload the exit code,
cycles and what it ran on; exit 1 if any failed."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def layer_suite():
    """validation/layer/simulate_shapes.py as a module of its own name, so it never shadows this file."""
    path = os.path.join(os.path.dirname(HERE), "layer", "simulate_shapes.py")
    spec = importlib.util.spec_from_file_location("layer_simulate_shapes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


layer = layer_suite()
KEYS, CI_TOKENS, sim = layer.KEYS, layer.CI_TOKENS, layer.sim
tokens, token_pairs = layer.tokens, layer.token_pairs


def under_cap(w):
    """Whether --ci times a workload, and so whether PyTorchSim's CI bundle holds it: a layer under the layer suite's
    cap. The ops alone are left to --where section=op on a bundle that holds them."""
    return w["section"] == "layer" and tokens(w) <= CI_TOKENS


if __name__ == "__main__":
    sys.exit(sim.main(here=HERE, keys=KEYS, doc=__doc__, rank=token_pairs, in_ci=under_cap))
