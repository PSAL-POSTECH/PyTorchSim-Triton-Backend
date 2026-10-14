"""Where the simulators are installed, and overrides (unset by default) of what a kernel object
says to run under (as_compiled.py)."""
import os

from pytorchsim_triton_compiler.toolchain import TOGSIM_DIR


def _env(name, default):
    """The environment variable, else `default`."""
    got = os.environ.get(name)
    return default if got is None else got


# The fork of Spike that loads the model, and pk, in the vcix-accelerator environment
# (VCIX_ENV_ROOT). Spike keeps an override for a hand-built fork (the fp8 one, a strict superset).
_VCIX_ENV = _env("VCIX_ENV_ROOT", "/opt/vcix-env")
SPIKE = _env("TORCHSIM_COMPILE_SPIKE", os.path.join(_VCIX_ENV, "spike", "bin", "spike"))
PK = os.path.join(_VCIX_ENV, "pk", "riscv64-unknown-elf", "bin", "pk")

# Overrides, unset by default: Spike's ISA, the model library (a full path), the machine yaml.
SPIKE_ISA = os.environ.get("TORCHSIM_COMPILE_SPIKE_ISA") or None
VCIX_MODEL = os.environ.get("TORCHSIM_COMPILE_VCIX_MODEL") or None
MACHINE_CONFIG = os.environ.get("TORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG") or None

# Where the models live (the kernel object's accelerator names which), and the gem5 script that
# times every model.
VCIX_BUILD = os.path.join(_env("TORCHSIM_PREFIX", "/workspace"), "vcix-build")
GEM5_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "third_party", "vcix-accelerator", "examples", "tpu", "gem5", "script_systolic.py")

# gem5, and TOGSim: the compiler's TOGSIM_DIR (third_party/TOGSim, built by setup/togsim.sh), whose
# machine files the targets are.
GEM5 = os.path.join(_VCIX_ENV, "gem5", "build", "RISCV", "gem5.opt")
TOGSIM = os.path.join(TOGSIM_DIR, "build", "bin", "Simulator")
