"""Where every external tool is, and whether this box can actually run one."""
import os
import re
import shutil
import subprocess
import sys

from . import machine
from .contract.target import TargetError


def _env(name, default):
    got = os.environ.get(name)
    if got is not None:
        return got
    return default() if callable(default) else default


PREFIX = _env("TORCHSIM_PREFIX", "/workspace")
VCIX_ENV = _env("VCIX_ENV_ROOT", "/opt/vcix-env")

LLVM_BIN = os.path.join(PREFIX, "LLVM_DIR", "llvm-project", "build", "install", "bin")
MLIR_TRANSLATE = os.path.join(LLVM_BIN, "mlir-translate")
LLC = os.path.join(LLVM_BIN, "llc")
OPT = os.path.join(LLVM_BIN, "opt")
MLIR_PYTHON_PACKAGES = os.path.join(os.path.dirname(LLVM_BIN), "python_packages", "mlir_core")

TRITON_PYTHON = os.path.join(PREFIX, "triton-src", "python")
TRITON_EXT = _env("TORCHSIM_TRITON_EXT",
                  os.path.join(PREFIX, "torchsim-triton-ext-build", "libtorchsim_triton_ext.so"))


def triton_env():
    """The environment the triton-compile worker runs in: that Triton first on the path, its
    backends found in its own tree (where the extension's `torchsim` backend is linked) rather than
    through whatever other Triton's entry points are installed, the extension loaded."""
    env = dict(os.environ)
    env["PYTHONPATH"] = TRITON_PYTHON + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TRITON_BACKENDS_IN_TREE"] = "1"
    env["TRITON_PLUGIN_PATHS"] = TRITON_EXT
    return env



def _togsim_dir():
    """TOGSIM_DIR, the root of a TOGSim checkout, else the third_party/TOGSim submodule. The retired
    name is refused, not aliased: it named a PyTorchSim checkout, one directory above this one."""
    if "TORCHSIM_COMPILE_TOGSIM_DIR" in os.environ:
        raise SystemExit(
            "TORCHSIM_COMPILE_TOGSIM_DIR is retired: it named a PyTorchSim checkout. Unset it; to "
            "use a TOGSim other than third_party/TOGSim, set TOGSIM_DIR to the root of that "
            "checkout (header $TOGSIM_DIR/include, binary $TOGSIM_DIR/build/bin/Simulator).")
    return _env("TOGSIM_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "TOGSim"))


# The TOGSim checkout a TOG is compiled against: the third_party/TOGSim submodule this
# checkout pins, so the TOG and the header it calls move together. A caller that
# replays the TOG on another TOGSim names that one.
TOGSIM_DIR = _togsim_dir()
TOGSIM_INCLUDE = os.path.join(TOGSIM_DIR, "include")
HOST_CXX = _env("CXX", "g++")

RISCV_GCC = os.path.join(VCIX_ENV, "riscv", "bin", "riscv64-unknown-elf-gcc")
RISCV_OBJDUMP = os.path.join(VCIX_ENV, "riscv", "bin", "riscv64-unknown-elf-objdump")
RISCV_READELF = os.path.join(VCIX_ENV, "riscv", "bin", "riscv64-unknown-elf-readelf")
RISCV_NM = os.path.join(VCIX_ENV, "riscv", "bin", "riscv64-unknown-elf-nm")


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
    """Where `import mlir` ACTUALLY resolved."""
    mlir = sys.modules.get("mlir")
    return list(getattr(mlir, "__path__", [])) if mlir else []


def foreign_bindings():
    """Loaded binding roots that are NOT the configured LLVM's."""
    keep = os.path.realpath(MLIR_PYTHON_PACKAGES)
    return [p for p in loaded_bindings()
            if not os.path.realpath(p).startswith(keep)]


def assert_bindings_clean():
    """Raise if another LLVM's bindings got mixed in."""
    bad = foreign_bindings()
    if bad:
        raise RuntimeError(
            "MLIR python bindings are mixed across LLVM versions.\n"
            f"  configured : {MLIR_PYTHON_PACKAGES}\n"
            f"  also loaded: {bad}\n"
            "`mlir` is a namespace package, so these silently merge. Something "
            "re-inserted a foreign mlir_core into sys.path after pytorchsim-triton-compiler selected "
            "one -- look for a hardcoded bindings path at module scope.")


def ensure_bindings():
    """Put LLVM_BIN's bindings first, import them, and refuse a mix -- what every in-process MLIR
    step does before it parses."""
    activate_bindings(sys.path)
    import mlir  # noqa: F401
    assert_bindings_clean()


def _which(path):
    if os.path.isabs(path):
        return path if os.path.isfile(path) else None
    return shutil.which(path)


def _version(path):
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
    ("OPT", OPT),
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
          f"vlen {t.vlen_bits} b, accelerator {t.accel}")
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
    dialect = _report_dialect_build()
    ok = dialect and ok
    ok = _check_triton_backend() and ok
    _check_togsim_header()
    if dialect:
        ok = _check_vcix_ops() and ok
    else:
        print(f"{'VCIX_OPS':<20} {'SKIPPED':<8} needs the dialect build above")
    print("NOTE: this harness never invokes mlir-opt -- every pass runs in-process")
    print("      through the Python bindings, so no fork of mlir-opt is needed --")
    print("      but the VCIX ops above are a fork's, and stage 5 mints them.")
    return ok


def _report_dialect_build():
    try:
        from .dialects.torchsim import BUILD
    except ImportError as e:
        print(f"{'TORCHSIM_DIALECT':<20} {'BAD':<8} {e}")
        return False
    print(f"{'TORCHSIM_DIALECT':<20} {'ok':<8} {BUILD}")
    print(f"{'':<29} stamp matches this checkout's .td")
    return True


def _check_triton_backend():
    import subprocess
    probe = ("import sys; sys.path.insert(0, %r); "
             "from pytorchsim_triton_compiler.lowering.triton_compile import backend_available; "
             "ok, said = backend_available(); print(said); sys.exit(0 if ok else 1)"
             % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    got = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         env=triton_env())
    ok, said = got.returncode == 0, (got.stdout.strip() or got.stderr.strip()[-200:])
    print(f"{'TRITON_BACKEND':<20} {'ok' if ok else 'BAD':<8} {said}")
    return ok


def togsim_api_version():
    """The TOGSim API version TOGSIM_INCLUDE/togsim.hpp states (its API_VERSION), or None when the
    header or the constant is missing."""
    try:
        text = open(os.path.join(TOGSIM_INCLUDE, "togsim.hpp")).read()
    except OSError:
        return None
    ver = re.search(r'inline constexpr char API_VERSION\[\] = "([^"]+)";', text)
    return ver.group(1) if ver else None


def _check_togsim_header():
    path = os.path.join(TOGSIM_INCLUDE, "togsim.hpp")
    cxx = _which(HOST_CXX)
    ver = togsim_api_version()
    if ver is None:
        print(f"{'TOGSIM_HEADER':<20} {'MISSING':<8} {path} (or its API_VERSION)")
        print(f"{'':<29} only --tog needs it (tog.so is compiled against it)")
    else:
        print(f"{'TOGSIM_HEADER':<20} {'ok':<8} {path}")
        print(f"{'':<29} TOGSim API v{ver}")
    print(f"{'HOST_CXX':<20} {'ok' if cxx else 'MISSING':<8} {cxx or HOST_CXX}")




def _check_vcix_ops():
    try:
        import mlir.ir as _ir
        from .lowering.passes.lib_vcix import VCIX_OPS, unregistered_ops
    except ImportError as e:
        print(f"{'VCIX_OPS':<20} {'FAIL':<8} {e}")
        return False
    missing = unregistered_ops(_ir.Context())
    print(f"{'VCIX_OPS':<20} {'MISSING' if missing else 'ok':<8} "
          f"{', '.join(missing or VCIX_OPS)}")
    if missing:
        print(f"{'':<29} this is not the LLVM pinned in setup/versions.env, whose branch carries them")
    return not missing
