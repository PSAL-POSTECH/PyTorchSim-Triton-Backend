"""Where every external tool is, and whether this box can actually run one."""
import os
import re
import shutil
import subprocess
import sys

from . import machine
from .contract.target import TargetError


# WHERE EACH TOOL IS, read once at import, so changing the environment later does
# not move one.
def _env(name, default):
    """The environment variable, else `default` -- CALLED if it is a callable."""
    got = os.environ.get(name)
    if got is not None:
        return got
    return default() if callable(default) else default


# TWO ROOTS, and every location follows from them: the tree setup/toolchain.sh lays
# (TORCHSIM_PREFIX) and the vcix-accelerator environment it sits on (VCIX_ENV_ROOT). The
# names under each are the script's; nothing is placed by hand.
PREFIX = _env("TORCHSIM_PREFIX", "/workspace")
VCIX_ENV = _env("VCIX_ENV_ROOT", "/opt/vcix-env")

LLVM_BIN = os.path.join(PREFIX, "LLVM_DIR", "llvm-project", "build", "install", "bin")
MLIR_TRANSLATE = os.path.join(LLVM_BIN, "mlir-translate")
LLC = os.path.join(LLVM_BIN, "llc")
MLIR_PYTHON_PACKAGES = os.path.join(os.path.dirname(LLVM_BIN), "python_packages", "mlir_core")

# Stages 1 and 2 run on an unmodified Triton (its python/ tree, libtriton.so in it) with the
# torchsim extension plugged in; everything this route adds to Triton lives in the extension.
TRITON_PYTHON = os.path.join(PREFIX, "triton-src", "python")
TRITON_EXT = os.path.join(PREFIX, "torchsim-triton-ext-build", "libtorchsim_triton_ext.so")


def triton_env():
    """The environment a stage-1 or stage-2 worker runs in: that Triton first on the path, its
    backends found in its own tree (where the extension's `torchsim` backend is linked) rather
    than through whatever other Triton's entry points are installed, the extension loaded."""
    env = dict(os.environ)
    env["PYTHONPATH"] = TRITON_PYTHON + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TRITON_BACKENDS_IN_TREE"] = "1"
    env["TRITON_PLUGIN_PATHS"] = TRITON_EXT
    return env


# The model Spike loads is the one thing beside the tree somebody swaps (another example,
# a build of their own), so it alone keeps an override.
VCIX_MODEL = _env("TORCHSIM_COMPILE_VCIX_MODEL", os.path.join(PREFIX, "vcix-build", "libtpu.so"))

# TOGSim and its machine configs: two directories of the PyTorchSim fork, nothing else of it.
# The config is a choice of machine, so it is overridable; the rest is where the fork is.
TOGSIM_DIR = os.path.join(PREFIX, "PyTorchSim")
TOGSIM_INCLUDE = os.path.join(TOGSIM_DIR, "TOGSim", "include")
TOGSIM = os.path.join(TOGSIM_DIR, "TOGSim", "build", "bin", "Simulator")
TOGSIM_CONFIG = _env("TORCHSIM_COMPILE_TOGSIM_CONFIG", os.path.join(
    TOGSIM_DIR, "configs", "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"))
GEM5 = os.path.join(VCIX_ENV, "gem5", "build", "RISCV", "gem5.opt")
# The reference machine lives beside the model it loads, in the submodule.
GEM5_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "vcix-accelerator", "examples", "tpu", "gem5", "script_systolic.py")
HOST_CXX = _env("CXX", "g++")

RISCV_GCC = os.path.join(VCIX_ENV, "riscv", "bin", "riscv64-unknown-elf-gcc")
RISCV_OBJDUMP = os.path.join(VCIX_ENV, "riscv", "bin", "riscv64-unknown-elf-objdump")
RISCV_READELF = os.path.join(VCIX_ENV, "riscv", "bin", "riscv64-unknown-elf-readelf")
RISCV_NM = os.path.join(VCIX_ENV, "riscv", "bin", "riscv64-unknown-elf-nm")


# WHICH MLIR THIS PROCESS LOADED -- not the same question as where it is.
def activate_bindings(sys_path):
    """Make `import mlir` resolve to LLVM_BIN's bindings, and ONLY those."""
    keep = os.path.realpath(MLIR_PYTHON_PACKAGES)
    foreign = [p for p in sys_path
               if p and os.path.realpath(p) != keep
               and os.path.isdir(os.path.join(p, "mlir", "_mlir_libs"))]
    for p in foreign:
        sys_path.remove(p)
    if sys_path[:1] != [MLIR_PYTHON_PACKAGES]:
        if MLIR_PYTHON_PACKAGES in sys_path:
            sys_path.remove(MLIR_PYTHON_PACKAGES)
        sys_path.insert(0, MLIR_PYTHON_PACKAGES)
    return foreign


def loaded_bindings():
    """Where `import mlir` ACTUALLY resolved. Returns [] if never imported."""
    mlir = sys.modules.get("mlir")
    return list(getattr(mlir, "__path__", [])) if mlir else []


def foreign_bindings():
    """Loaded binding roots that are NOT the configured LLVM's. Empty is correct."""
    keep = os.path.realpath(MLIR_PYTHON_PACKAGES)
    return [p for p in loaded_bindings()
            if not os.path.realpath(p).startswith(keep)]


def assert_bindings_clean():
    """Raise if another LLVM's bindings got mixed in. Cheap; call after import."""
    bad = foreign_bindings()
    if bad:
        raise RuntimeError(
            "MLIR python bindings are mixed across LLVM versions.\n"
            f"  configured : {MLIR_PYTHON_PACKAGES}\n"
            f"  also loaded: {bad}\n"
            "`mlir` is a namespace package, so these silently merge. Something "
            "re-inserted a foreign mlir_core into sys.path after pytorchsim-triton-compiler selected "
            "one -- look for a hardcoded bindings path at module scope.")


# THE REPORT: `doctor`, its table and the checks it makes, in the order it makes them.
def _which(path):
    """Resolve a bare command name via PATH; return absolute path or None."""
    if os.path.isabs(path):
        return path if os.path.isfile(path) else None
    return shutil.which(path)




def _version(path):
    """The line a tool prints about its version, one run."""
    try:
        out = subprocess.run([path, "--version"], capture_output=True,
                             text=True, timeout=20)
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}>"
    text = out.stdout + out.stderr
    return next((l.strip() for l in text.splitlines() if "version" in l.lower()),
                (text.strip().splitlines() or [""])[0][:80])


TOOLS = (
    ("MLIR_TRANSLATE", MLIR_TRANSLATE),
    ("LLC", LLC),
    ("RISCV_GCC", RISCV_GCC),
)


def doctor():
    """Print the resolved toolchain and return True if everything is usable."""
    ok = True
    print(f"{'component':<20} {'status':<8} path / version")
    print("-" * 100)
    for name, path in TOOLS:
        resolved = _which(path)
        if resolved is None:
            ok = False
            print(f"{name:<20} {'MISSING':<8} {path}")
            continue
        said = _version(resolved)
        print(f"{name:<20} {'ok':<8} {resolved}")
        print(f"{'':<29} {said}")
    for name, path in (("TRITON_PYTHON", TRITON_PYTHON), ("TRITON_EXT", TRITON_EXT)):
        there = os.path.exists(path)
        ok = ok and there
        print(f"{name:<20} {'ok' if there else 'MISSING':<8} {path}")

    print("-" * 100)
    try:
        t = machine.target()
    except TargetError as e:
        ok = False
        print(f"{'TARGET':<20} {'BAD':<8} {e}")
        return ok
    print(f"{'TARGET':<20} {'ok':<8} {t.name}")
    print(f"{'':<29} {t.origin}")
    print(f"{'':<29} {t.vector_lanes} lanes x {t.spad_size} B/lane, "
          f"vlen {t.vlen_bits} b")
    problems = machine.check_memory_map()
    span = f"0x{machine.SPAD_VADDR:x}-0x{machine.SPAD_VADDR + machine.banked_bytes():x}"
    if problems:
        ok = False
        print(f"{'SPAD_MAP':<20} {'BAD':<8} banked window {span} "
              f"({machine.VECTORLANE_SIZE} lanes x {machine.SPAD_SIZE} B)")
        for p in problems:
            print(f"{'':<29} {p}")
    else:
        print(f"{'SPAD_MAP':<20} {'ok':<8} banked window {span} "
              f"({machine.VECTORLANE_SIZE} lanes x {machine.SPAD_SIZE} B) "
              f"-> paddr 0x{machine.SPAD_PADDR:x}")

    if os.path.isdir(MLIR_PYTHON_PACKAGES):
        print(f"{'MLIR_BINDINGS':<20} {'ok':<8} {MLIR_PYTHON_PACKAGES}")
    else:
        ok = False
        print(f"{'MLIR_BINDINGS':<20} {'MISSING':<8} {MLIR_PYTHON_PACKAGES}")

    try:
        import mlir  # noqa: F401
        bad = foreign_bindings()
        if bad:
            ok = False
            print(f"{'MLIR_LOADED':<20} {'MIXED':<8} also loaded: {bad}")
            print(f"{'':<29} `mlir` is a namespace package -- these merge silently")
        else:
            print(f"{'MLIR_LOADED':<20} {'ok':<8} {loaded_bindings()[0]}")
    except ImportError as e:
        ok = False
        print(f"{'MLIR_LOADED':<20} {'FAIL':<8} {e}")

    print()
    _report_dialect_build()
    ok = _check_triton_backend() and ok
    _check_togsim_header()
    ok = _check_vcix_ops() and ok
    _check_vcix_model()
    print("NOTE: this harness never invokes mlir-opt -- every pass runs in-process")
    print("      through the Python bindings, so no fork of mlir-opt is needed --")
    print("      but the VCIX ops above are a fork's, and stage 5 mints them.")
    return ok


def _report_dialect_build():
    """WHICH dialect build this process loaded, which nothing else here says. Nothing to
    judge: the op classes are tablegen output and `dialects.torchsim` refuses a build made
    from another .td AT IMPORT, so reaching this line already means the stamp matched."""
    from .dialects.torchsim import BUILD
    print(f"{'TORCHSIM_DIALECT':<20} {'ok':<8} {BUILD}")
    print(f"{'':<29} stamp matches this checkout's .td")


def _check_triton_backend():
    """The `cpu` target stage 1 asks for resolves to the extension's backend, in the environment
    the stage-1 worker runs in, or it does not."""
    import subprocess
    probe = ("import sys; sys.path.insert(0, %r); "
             "from pytorchsim_triton_compiler.lowering.triton_ttir import backend_available; "
             "ok, said = backend_available(); print(said); sys.exit(0 if ok else 1)"
             % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    got = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         env=triton_env())
    ok, said = got.returncode == 0, (got.stdout.strip() or got.stderr.strip()[-200:])
    print(f"{'TRITON_BACKEND':<20} {'ok' if ok else 'BAD':<8} {said}")
    return ok


def _check_togsim_header():
    """The trace producer is compiled against TOGSim's runtime header. Never fatal:
    only the timing route needs it."""
    import os as _os
    path = _os.path.join(TOGSIM_INCLUDE, "togsim_runtime.h")
    cxx = _which(HOST_CXX)
    if not _os.path.isfile(path):
        print(f"{'TOGSIM_HEADER':<20} {'MISSING':<8} {path}")
        print(f"{'':<29} only --trace needs it (the trace producer's .so)")
    else:
        ver = re.search(r"#define\s+TOGSIM_ABI_VERSION\s+(\d+)", open(path).read())
        print(f"{'TOGSIM_HEADER':<20} {'ok':<8} {path}")
        print(f"{'':<29} ABI v{ver.group(1) if ver else '?'}")
    print(f"{'HOST_CXX':<20} {'ok' if cxx else 'MISSING':<8} {cxx or HOST_CXX}")
    for name, path in (("TOGSIM", TOGSIM), ("TOGSIM_CONFIG", TOGSIM_CONFIG),
                       ("GEM5", GEM5), ("GEM5_SCRIPT", GEM5_SCRIPT)):
        there = _os.path.isfile(path)
        print(f"{name:<20} {'ok' if there else 'MISSING':<8} {path}")
    print(f"{'':<29} only --simulate needs these")




def _check_vcix_model():
    """The model Spike loads. Nothing reads it yet, so this does not fail."""
    import os as _os
    there = _os.path.isfile(VCIX_MODEL)
    print(f"{'VCIX_MODEL':<20} {'ok' if there else 'MISSING':<8} {VCIX_MODEL}")
    if not there:
        print(f"{'':<29} `setup/toolchain.sh vcix` builds it from the submodule")
    return True


def _check_vcix_ops():
    """The VCIX ops lower_to_vcix and lower_to_xlu mint must be REGISTERED in LLVM_BIN's MLIR,
    and on a stock build none of them are."""
    import mlir.ir as _ir
    from .lowering.passes.lib_vcix import VCIX_OPS, unregistered_ops
    missing = unregistered_ops(_ir.Context())
    print(f"{'VCIX_OPS':<20} {'MISSING' if missing else 'ok':<8} "
          f"{', '.join(missing or VCIX_OPS)}")
    if missing:
        print(f"{'':<29} this is not the LLVM pinned in setup/versions.env, whose branch carries them")
    return not missing
