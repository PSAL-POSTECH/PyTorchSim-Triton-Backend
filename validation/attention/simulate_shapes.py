#!/usr/bin/env python3
"""Generate this suite's kernels under the tiling policy and time them on --target with TOGSim exactly as
validation/gemm does (its runner, run here with kind, Hq, Hkv and S naming a shape, ranked by shapes.csv's cost);
--ci times only the shapes under emit_kernels.CI_CAP. results.json holds the machine and per shape the exit code,
cycles and what it ran on; exit 1 if any failed."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("kind", "Hq", "Hkv", "S")


def gemm_runner():
    """validation/gemm/simulate_shapes.py as a module of its own name, so it never shadows this file."""
    path = os.path.join(os.path.dirname(HERE), "gemm", "simulate_shapes.py")
    spec = importlib.util.spec_from_file_location("gemm_simulate_shapes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sim = gemm_runner()

if __name__ == "__main__":
    sys.exit(sim.main(here=HERE, keys=KEYS, doc=__doc__))
