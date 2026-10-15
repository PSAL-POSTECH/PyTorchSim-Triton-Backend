#!/usr/bin/env python3
"""Time every shape of shapes.csv on --target with TOGSim exactly as validation/gemm does (its runner, run here
over kernels/ with program, R and C naming a shape): a binary, gem5 tile samples and a TOGSim run, --jobs at a
time; results.json holds the machine and per shape the exit code, cycles and what it ran on; exit 1 if any failed."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("program", "R", "C")


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
