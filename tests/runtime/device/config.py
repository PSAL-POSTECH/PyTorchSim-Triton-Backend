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


#
# This is the plain SPIKE_ROOT install, not the fp8 fork's build. The fork is a
# strict superset -- same base commit (ad85c5a), Zvfp8 on top -- but it is built
# by hand into a worktree that restore.sh does not create, so defaulting to it
# made a stock restore report SPIKE MISSING and stop stage 6 dead. The extension
# buys nothing yet either: fp8 is BLOCKED in stage 3, before spike is reached,
# because LLVM has no lowering for it (see tests/cases/mechanism/ops/ops_fp8_roundtrip
# .py). Point TORCHSIM_COMPILE_SPIKE and TORCHSIM_COMPILE_SPIKE_ISA at the fork the day that lifts.
SPIKE = _env("TORCHSIM_COMPILE_SPIKE", "/workspace/riscv-isa-sim/install/bin/spike")
PK = _env("TORCHSIM_COMPILE_PK", "/workspace/riscv-pk/build/pk")

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

# The accelerator, as a library Spike dlopens, and an override for the machine
# YAML the launcher otherwise writes.
VCIX_MODEL = _env("TORCHSIM_COMPILE_VCIX_MODEL", "/workspace/vcix-build/libtpu.so")
MACHINE_CONFIG = _env("TORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG",
                      _env("TOGSIM_CONFIG", ""))
