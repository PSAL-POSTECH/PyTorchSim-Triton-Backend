"""Is the simulator this suite runs on usable, and does it speak the ISA the target compiles for?

The compiler's own `doctor` does not ask: it never runs a program. Reported by
tests/torchsim-compile doctor, for TORCHSIM_COMPILE_TARGET, alongside what the compiler reports.
"""
import os
import re
import shutil
import subprocess

from pytorchsim_triton_compiler import machine
from . import config


def _which(path):
    """Resolve a bare command name via PATH; absolute path or None."""
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    return shutil.which(path)


def doctor():
    """Print SPIKE, PK, what the target runs on and the ISA Spike is asked for. True if usable."""
    ok = True
    for name, path in (("SPIKE", config.SPIKE), ("PK", config.PK)):
        r = _which(path)
        print(f"{name:<20} {'ok' if r else 'MISSING':<8} {r or path}")
        ok = ok and bool(r)
    t = machine.target()
    script, togsim = t.gem5_script(), config.TOGSIM_CONFIG or t.togsim_config
    for name, path in (
            ("VCIX_MODEL", config.VCIX_MODEL or os.path.join(config.VCIX_BUILD, t.model_library())),
            ("GEM5", config.GEM5),
            ("GEM5_SCRIPT", script and os.path.join(config.GEM5_SCRIPTS, script)),
            ("TOGSIM", config.TOGSIM),
            ("TOGSIM_CONFIG", togsim and os.path.join(config.TOGSIM_CONFIGS, togsim))):
        there = bool(path) and os.path.isfile(path)
        print(f"{name:<20} {'ok' if there else 'MISSING':<8} {path or f'none for target {t.name}'}")
    print(f"{'':<29} gem5 and TOGSim: only --timing needs them")
    return _check_spike_isa(config.SPIKE_ISA or t.spike_isa()) and ok


def _check_spike_isa(isa):
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
        out = subprocess.run([resolved, f"--isa={isa}",
                              "-m0x80000000:0x1000", "/nonexistent-elf"],
                             capture_output=True, text=True, timeout=20)
    except Exception as e:                 # noqa: BLE001 - diagnostics only
        print(f"{'SPIKE_ISA':<20} {'?':<8} could not probe ({type(e).__name__})")
        return True
    text = out.stdout + out.stderr
    if "bad --isa" in text or "unsupported extension" in text:
        why = next((l.strip() for l in text.splitlines() if "isa" in l.lower()),
                   text.strip().splitlines()[0] if text.strip() else "")
        print(f"{'SPIKE_ISA':<20} {'BAD':<8} {isa}")
        print(f"{'':<29} {why[:80]}")
        print(f"{'':<29} this spike does not have that extension, and it will")
        print(f"{'':<29} refuse to start -- so EVERY kernel compiled for this target")
        print(f"{'':<29} fails. Point TORCHSIM_COMPILE_SPIKE at a Spike that has it.")
        return False
    print(f"{'SPIKE_ISA':<20} {'ok':<8} {isa}")
    return True


_LLVM_VER_RE = re.compile(r"LLVM version\s+(\d+)")
