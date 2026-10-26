"""Where the simulators are installed, and overrides (unset by default) of what a kernel object
says to run under (as_compiled.py)."""
import os

from pytorchsim_triton_compiler import toolchain
from pytorchsim_triton_compiler.toolchain import TOGSIM_DIR


def _env(name, default):
    """The environment variable, else `default`."""
    got = os.environ.get(name)
    return default if got is None else got


# The fork of Spike that loads the model, and pk, in the vcix-accelerator environment
# (VCIX_ENV_ROOT). PYTORCHSIM_COMPILE_SPIKE overrides Spike with another build.
_VCIX_ENV = _env("VCIX_ENV_ROOT", "/opt/vcix-env")
SPIKE = _env("PYTORCHSIM_COMPILE_SPIKE", os.path.join(_VCIX_ENV, "spike", "bin", "spike"))
PK = os.path.join(_VCIX_ENV, "pk", "riscv64-unknown-elf", "bin", "pk")

# Overrides, unset by default: Spike's ISA, the model library (a full path), the machine yaml.
SPIKE_ISA = os.environ.get("PYTORCHSIM_COMPILE_SPIKE_ISA") or None
VCIX_MODEL = os.environ.get("PYTORCHSIM_COMPILE_VCIX_MODEL") or None
MACHINE_CONFIG = os.environ.get("PYTORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG") or None

# Where the models live (the kernel object's accelerator names which; contract.vcix_model also names
# the gem5 machine each is timed on).
VCIX_BUILD = toolchain.VCIX_BUILD

# gem5, and TOGSim: the compiler's TOGSIM_DIR (third_party/TOGSim, built by setup/togsim.sh), whose
# machine files the targets are.
GEM5 = toolchain.GEM5
TOGSIM = os.path.join(TOGSIM_DIR, "build", "bin", "Simulator")
