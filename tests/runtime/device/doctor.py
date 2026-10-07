"""Is the simulator this suite runs on usable, and does it speak the right ISA?

The compiler's own `doctor` does not ask: it never runs a program. Reported by
tests/torchsim-compile doctor, alongside what the compiler reports.
"""
import os
import re
import shutil
import subprocess

from . import config


def _which(path):
    """Resolve a bare command name via PATH; absolute path or None."""
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    return shutil.which(path)


def doctor():
    """Print SPIKE, PK and the ISA they were asked for. True if usable."""
    ok = True
    for name, path in (("SPIKE", config.SPIKE), ("PK", config.PK)):
        r = _which(path)
        print(f"{name:<20} {'ok' if r else 'MISSING':<8} {r or path}")
        ok = ok and bool(r)
    for name, path in (("VCIX_MODEL", config.VCIX_MODEL), ("GEM5", config.GEM5),
                       ("GEM5_SCRIPT", config.GEM5_SCRIPT), ("TOGSIM", config.TOGSIM),
                       ("TOGSIM_CONFIG", config.TOGSIM_CONFIG)):
        there = os.path.isfile(path)
        print(f"{name:<20} {'ok' if there else 'MISSING':<8} {path}")
    print(f"{'':<29} gem5 and TOGSim: only --simulate needs them")
    return _check_spike_isa() and ok


def _check_spike_isa():
    """Does the configured Spike actually accept the configured ISA string?

    Worth a subprocess because of HOW this fails. Spike parses --isa before it
    opens the ELF, so a build without Zvfp8 does not fail at the first fp8
    instruction in an fp8 kernel -- it refuses to start, and EVERY kernel dies
    in stage 6 with the same opaque non-zero exit. The one line that says why
    goes to stderr ahead of any output the harness reads.

    The probe is that same launch with a file that cannot exist: reaching "could
    not open" means the ISA was accepted, which is the whole question.
    """
    resolved = _which(config.SPIKE)
    if resolved is None:
        return True                        # already reported MISSING above
    try:
        out = subprocess.run([resolved, f"--isa={config.SPIKE_ISA}",
                              "-m0x80000000:0x1000", "/nonexistent-elf"],
                             capture_output=True, text=True, timeout=20)
    except Exception as e:                 # noqa: BLE001 - diagnostics only
        print(f"{'SPIKE_ISA':<20} {'?':<8} could not probe ({type(e).__name__})")
        return True
    text = out.stdout + out.stderr
    if "bad --isa" in text or "unsupported extension" in text:
        why = next((l.strip() for l in text.splitlines() if "isa" in l.lower()),
                   text.strip().splitlines()[0] if text.strip() else "")
        print(f"{'SPIKE_ISA':<20} {'BAD':<8} {config.SPIKE_ISA}")
        print(f"{'':<29} {why[:80]}")
        print(f"{'':<29} this spike does not have that extension, and it will")
        print(f"{'':<29} refuse to start -- so EVERY kernel fails, not just the")
        print(f"{'':<29} fp8 ones. Either point TORCHSIM_COMPILE_SPIKE at the fp8 fork's")
        print(f"{'':<29} build, or drop _zvfp8 from TORCHSIM_COMPILE_SPIKE_ISA and accept")
        print(f"{'':<29} that fp8 kernels cannot run.")
        return False
    print(f"{'SPIKE_ISA':<20} {'ok':<8} {config.SPIKE_ISA}")
    return True


_LLVM_VER_RE = re.compile(r"LLVM version\s+(\d+)")
