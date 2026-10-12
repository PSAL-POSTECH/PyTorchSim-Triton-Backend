"""Where the simulators are installed, and what ISA to ask Spike for.

The compiler does not carry these among its own tools: it never runs the ELF it
produces. Machine numbers are NOT here either -- those come from the kernel
object's target.
"""
import os


def _env(name, default):
    """The environment variable, else `default`."""
    got = os.environ.get(name)
    return default if got is None else got


# The fork of Spike that loads the model, and pk, in the vcix-accelerator environment
# (VCIX_ENV_ROOT). Spike alone keeps an override: the fp8 fork's build is a strict superset
# (same base commit, Zvfp8 on top) built by hand, and fp8 is BLOCKED in stage 3 before spike
# is reached anyway, because LLVM has no lowering for it (tests/cases/mechanism/ops/
# ops_fp8_roundtrip.py). Point TORCHSIM_COMPILE_SPIKE and _SPIKE_ISA at it the day that lifts.
_VCIX_ENV = _env("VCIX_ENV_ROOT", "/opt/vcix-env")
SPIKE = _env("TORCHSIM_COMPILE_SPIKE", os.path.join(_VCIX_ENV, "spike", "bin", "spike"))
PK = os.path.join(_VCIX_ENV, "pk", "riscv64-unknown-elf", "bin", "pk")

# The ISA string stage 6 launches with. Two things are load-bearing:
#   * zvfp8 is the fork's fp8 extension -- E4M3 and E5M2 share one `float8_t`
#     and the format is chosen at RUNTIME (DMA descriptor byte 117), so there is
#     one extension rather than two;
#   * zfh is required WITH it, not merely alongside: every widening op out of
#     e8 lands in e16 (vfwcvt.f.f.v, vfncvt.f.f.w, vfwredosum.vs) and traps
#     without it. The fork's own gating test asserts each of those traps.
# Enabling an extension only ADDS instructions, so naming zvfp8 is neutral for
# every kernel with no fp8 in it -- but it is fatal on a spike that lacks it,
# and fatal for the WHOLE SUITE rather than just the fp8 part: spike rejects the
# --isa string and exits before the ELF is opened, so add.py dies the same way
# an fp8 kernel would. That asymmetry is why the default stops at zfh and why
# doctor() probes the pair instead of trusting them to match: a missing
# extension costs one blocked kernel that stage 3 already blocks, while naming
# one the binary lacks costs everything.
# _xvcixaccel is the adapter: without it every instruction the model owns is
# illegal. The extension the paragraph above argues against naming, named.
SPIKE_ISA = _env("TORCHSIM_COMPILE_SPIKE_ISA", "rv64gcv_zfh_xvcixaccel")

# The accelerator, as a library Spike dlopens (the same default and override as the compiler's
# toolchain.py, restated here because the device never imports the compiler), and an override
# for the machine YAML the launcher otherwise writes.
VCIX_MODEL = _env("TORCHSIM_COMPILE_VCIX_MODEL",
                  os.path.join(_env("TORCHSIM_PREFIX", "/workspace"), "vcix-build", "libtpu.so"))
MACHINE_CONFIG = _env("TORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG", "")

# gem5 from the same environment, the vcix-accelerator script it runs a sample tile under, and
# TOGSim out of the third_party/PyTorchSim submodule, built in its tree by setup/togsim.sh (the same
# default and override as the compiler's toolchain.py).
GEM5 = os.path.join(_VCIX_ENV, "gem5", "build", "RISCV", "gem5.opt")
GEM5_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "third_party", "vcix-accelerator", "examples", "tpu", "gem5", "script_systolic.py")
TOGSIM_DIR = _env("TORCHSIM_COMPILE_TOGSIM_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "third_party", "PyTorchSim"))
TOGSIM = os.path.join(TOGSIM_DIR, "TOGSim", "build", "bin", "Simulator")
TOGSIM_CONFIG = _env("TORCHSIM_COMPILE_TOGSIM_CONFIG", os.path.join(
    TOGSIM_DIR, "configs", "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"))
