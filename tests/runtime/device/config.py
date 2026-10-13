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
# an fp8 kernel would. That asymmetry is why the default names only what the pinned
# spike has (zfh, zvfbfmin, zvfofp8min) and why
# doctor() probes the pair instead of trusting them to match: a missing
# extension costs one blocked kernel that stage 3 already blocks, while naming
# one the binary lacks costs everything.
# _xvcixaccel is the adapter: without it every instruction the model owns is
# illegal. The extension the paragraph above argues against naming, named.
SPIKE_ISA = _env("TORCHSIM_COMPILE_SPIKE_ISA", "rv64gcv_zfh_zvfbfmin_zvfofp8min_xvcixaccel")

# The accelerator, as a library Spike and gem5 dlopen: the one whose unit is the target's
# matmul_unit (SYSTOLIC of libtpu.so, TPUV6_MXU of libtpuv6.so), from setup/toolchain.sh's vcix
# step, unless TORCHSIM_COMPILE_VCIX_MODEL names one for every target. And an override for the
# machine YAML the launcher otherwise writes.
VCIX_MODELS = {unit: os.path.join(_env("TORCHSIM_PREFIX", "/workspace"), "vcix-build", so)
               for unit, so in (("systolic", "libtpu.so"), ("tpuv6_mxu", "libtpuv6.so"))}


def vcix_model(target):
    """The model library for `target`: the override, else the one built for its matmul_unit."""
    return _env("TORCHSIM_COMPILE_VCIX_MODEL", VCIX_MODELS[target.matmul_unit])


MACHINE_CONFIG = _env("TORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG", "")

# gem5 from the same environment, the vcix-accelerator script it runs a sample tile under, and
# TOGSim, the third_party/TOGSim submodule, built in its tree by setup/togsim.sh (the same
# default and override as the compiler's toolchain.py).
GEM5 = os.path.join(_VCIX_ENV, "gem5", "build", "RISCV", "gem5.opt")
GEM5_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "third_party", "vcix-accelerator", "examples", "tpu", "gem5", "script_systolic.py")
TOGSIM_DIR = _togsim_dir()
TOGSIM = os.path.join(TOGSIM_DIR, "build", "bin", "Simulator")
# TOGSim's description of the same machine as a target, by the target's name: a config is a whole
# machine (clock, DRAM, arrays), which a target's matmul_unit does not say.
TOGSIM_CONFIGS = {name: os.path.join(TOGSIM_DIR, "configs", yml) for name, yml in (
    ("tpuv6e", "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"),)}


def togsim_config(target):
    """The TOGSim config for `target`: TORCHSIM_COMPILE_TOGSIM_CONFIG, else the one of the machine
    the target is named for; a target of no machine known here is refused, not given another's."""
    said = os.environ.get("TORCHSIM_COMPILE_TOGSIM_CONFIG")
    if said is not None:
        return said
    if target.name not in TOGSIM_CONFIGS:
        raise ValueError(
            f"target {target.name!r} is no machine TOGSim has a config for here (known: "
            f"{', '.join(sorted(TOGSIM_CONFIGS))}); set TORCHSIM_COMPILE_TOGSIM_CONFIG to the "
            f"TOGSim config of that machine.")
    return TOGSIM_CONFIGS[target.name]
