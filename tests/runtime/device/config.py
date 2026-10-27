"""Where the simulators are installed, and overrides (unset by default) of what a kernel object
says to run under (as_compiled.py)."""
import os

from pytorchsim_triton_compiler import toolchain
from pytorchsim_triton_compiler.toolchain import TOGSIM_DIR


# Spike and pk, the compiler's (the Coprocessor environment, COPROCESSOR_ENV_ROOT).
SPIKE = toolchain.SPIKE
PK = toolchain.PK

# Overrides, unset by default: Spike's ISA, the model library (a full path), the machine yaml.
SPIKE_ISA = os.environ.get("PYTORCHSIM_COMPILE_SPIKE_ISA") or None
COPROCESSOR_MODEL = os.environ.get("PYTORCHSIM_COMPILE_COPROCESSOR_MODEL") or None
MACHINE_CONFIG = os.environ.get("PYTORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG") or None

# Where the models live (the kernel object's accelerator names which; contract.vcix_model also names
# the gem5 machine each is timed on).
COPROCESSOR_BUILD = toolchain.COPROCESSOR_BUILD

# gem5, and TOGSim: the compiler's TOGSIM_DIR (third_party/TOGSim, built by setup/togsim.sh), whose
# machine files the subtargets are.
GEM5 = toolchain.GEM5
TOGSIM = os.path.join(TOGSIM_DIR, "build", "bin", "Simulator")
