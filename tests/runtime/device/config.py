"""Where the simulators are installed, and the overrides of what a kernel object says to run
under -- each unset unless a caller sets it. The ISA, the model, the machine yaml and the TOGSim
config come from the kernel object (as_compiled.py), not from here."""
import os


def _env(name, default):
    """The environment variable, else `default`."""
    got = os.environ.get(name)
    return default if got is None else got


def _togsim_dir():
    """TOGSIM_DIR, the root of a TOGSim checkout, else the third_party/TOGSim submodule. The retired
    name is refused, not aliased: it named a PyTorchSim checkout, one directory above this one."""
    if "TORCHSIM_COMPILE_TOGSIM_DIR" in os.environ:
        raise SystemExit(
            "TORCHSIM_COMPILE_TOGSIM_DIR is retired: it named a PyTorchSim checkout. Unset it; to "
            "use a TOGSim other than third_party/TOGSim, set TOGSIM_DIR to the root of that "
            "checkout (header $TOGSIM_DIR/include, binary $TOGSIM_DIR/build/bin/Simulator).")
    return _env("TOGSIM_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "third_party", "TOGSim"))


# The fork of Spike that loads the model, and pk, in the vcix-accelerator environment
# (VCIX_ENV_ROOT). Spike keeps an override for a hand-built fork (the fp8 one, a strict superset).
_VCIX_ENV = _env("VCIX_ENV_ROOT", "/opt/vcix-env")
SPIKE = _env("TORCHSIM_COMPILE_SPIKE", os.path.join(_VCIX_ENV, "spike", "bin", "spike"))
PK = os.path.join(_VCIX_ENV, "pk", "riscv64-unknown-elf", "bin", "pk")

# Overrides, unset by default: Spike's ISA, the model library (a full path), the machine yaml.
SPIKE_ISA = os.environ.get("TORCHSIM_COMPILE_SPIKE_ISA") or None
VCIX_MODEL = os.environ.get("TORCHSIM_COMPILE_VCIX_MODEL") or None
MACHINE_CONFIG = os.environ.get("TORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG") or None

# Where the accelerator models are built (the kernel object names which one) and where the
# vcix-accelerator gem5 scripts are (the kernel object names which one).
VCIX_BUILD = os.path.join(_env("TORCHSIM_PREFIX", "/workspace"), "vcix-build")
GEM5_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "third_party", "vcix-accelerator", "examples", "tpu", "gem5")

# gem5 from the same environment, and TOGSim, the third_party/TOGSim submodule built in its tree by
# setup/togsim.sh; its config is the target's togsim_config unless overridden here (a path).
GEM5 = os.path.join(_VCIX_ENV, "gem5", "build", "RISCV", "gem5.opt")
TOGSIM_DIR = _togsim_dir()
TOGSIM = os.path.join(TOGSIM_DIR, "build", "bin", "Simulator")
TOGSIM_CONFIGS = os.path.join(TOGSIM_DIR, "configs")
TOGSIM_CONFIG = os.environ.get("TORCHSIM_COMPILE_TOGSIM_CONFIG") or None
