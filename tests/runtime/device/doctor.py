"""Is the simulator this suite runs on usable, and does it speak the ISA the target compiles for?

The compiler's own `doctor` does not ask: it never runs a program. Reported by
tests/torchsim-compile doctor, for TORCHSIM_COMPILE_TARGET, alongside what the compiler reports.
"""
import os
import re
import shutil
import subprocess

from pytorchsim_triton_compiler import machine
from pytorchsim_triton_compiler.contract import kernel_object, vcix_model
from . import config


def _which(path):
    """Resolve a bare command name via PATH; absolute path or None."""
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    return shutil.which(path)


def doctor():
    """Print SPIKE, PK, what the target runs on and the ISA Spike is asked for, as run_options answers
    them for a launch (no kernel object, so no machine yaml). True if usable."""
    ok = True
    for name, path in (("SPIKE", config.SPIKE), ("PK", config.PK)):
        r = _which(path)
        print(f"{name:<20} {'ok' if r else 'MISSING':<8} {r or path}")
        ok = ok and bool(r)
    t = machine.target()
    said = {"spike_isa": config.SPIKE_ISA, "model": config.VCIX_MODEL, "machine_config": "-"}
    try:
        run = kernel_object.run_options(t, None, config.VCIX_BUILD, **said)
    except vcix_model.ModelError as e:
        print(f"{'VCIX_MODEL':<20} {'MISSING':<8} {e}")
        run = dict(kernel_object.run_options(t, None, config.VCIX_BUILD, **dict(said, model="-")),
                   model=None)
    for name, path in (
            *((("VCIX_MODEL", run["model"]),) if run["model"] else ()),
            ("GEM5", config.GEM5),
            ("GEM5_SCRIPT", run["gem5_script"]),
            ("TOGSIM", config.TOGSIM),
            ("MACHINE_FILE", t.file)):
        there = bool(path) and os.path.isfile(path)
        print(f"{name:<20} {'ok' if there else 'MISSING':<8} {path or f'none for target {t.name}'}")
    print(f"{'':<29} gem5 and TOGSim: only --timing needs them")
    return _check_spike_isa(run["spike_isa"]) and ok


def _check_spike_isa(isa):
    """Whether Spike takes `isa`: it parses --isa before it opens the ELF, so a Spike without one of
    the target's extensions refuses every kernel at the spike stage with one opaque exit. Probed with
    an ELF that cannot exist; reaching "could not open" means the ISA was accepted."""
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
