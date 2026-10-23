"""Where every external tool is, and whether this box can actually run one."""
import os
import shutil
import subprocess
import sys
import tempfile

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
# simulates the TOG on another TOGSim names that one.
TOGSIM_DIR = _togsim_dir()
TOGSIM_INCLUDE = os.path.join(TOGSIM_DIR, "include")
HOST_CXX = _env("CXX", "g++")

RISCV_GCC = os.path.join(VCIX_ENV, "riscv", "bin", "riscv64-unknown-elf-gcc")
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


class ToolchainError(RuntimeError):
    pass


SPAD_VECTOR_MEMORY = "spad-vector-memory"
_LLC_HONOURS = {}


def _vector_copy_ir(space):
    """An <8 x i32> load and store through two pointers in address space `space`, as llc reads it."""
    ptr = f"ptr addrspace({space})"
    return (f"define void @witness({ptr} %src, {ptr} %dst) {{\n"
            f"  %v = load <8 x i32>, {ptr} %src, align 4\n"
            f"  store <8 x i32> %v, {ptr} %dst, align 4\n"
            f"  ret void\n}}\n")


def check_llc_honours(target):
    """When `target` asks for spad-vector-memory: LLC compiles a vector copy through SPAD_SPACE, refuses
    one through address space 0, and counts a forced vector spill in its regalloc remark; else
    ToolchainError naming LLC with what it printed. Once per (LLC, its mtime, -mattr)."""
    if SPAD_VECTOR_MEMORY not in target.isa:
        return
    from .lowering.passes.lib_bank_layout import SPAD_SPACE
    mattr = target.llc_mattr()
    try:
        key = (LLC, os.path.getmtime(LLC), mattr)
    except OSError as e:
        raise ToolchainError(f"{LLC}: {e.strerror}; the target asks llc for +{SPAD_VECTOR_MEMORY}")
    if key in _LLC_HONOURS:
        return
    argv = [LLC, f"-march={target.llc_march()}", f"-mattr={mattr}", "-O3", "-o", os.devnull, "-"]
    for space, compiles in ((SPAD_SPACE, True), (0, False)):
        got = subprocess.run(argv, input=_vector_copy_ir(space), capture_output=True, text=True,
                             timeout=120)
        if (got.returncode == 0) != compiles:
            raise ToolchainError(
                f"{LLC} does not honour +{SPAD_VECTOR_MEMORY}: an <8 x i32> load and store through "
                f"ptr addrspace({space}) exited {got.returncode}, and it must "
                + ("compile" if compiles else "be refused")
                + f"\n$ {' '.join(argv)}\n{got.stderr}")
    _check_llc_counts_vector_spills(target, SPAD_SPACE)
    _LLC_HONOURS[key] = True


def _vector_spill_ir(space, elems, loop_file):
    """Six <elems x i32> loads all live across six stores in reverse order: more vector registers than
    the machine has, so llc must spill vectors; every instruction at line 1 of `loop_file`, so the
    spills and reloads are placed in loop 1."""
    ty, ptr, at = f"<{elems} x i32>", f"ptr addrspace({space})", ", !dbg !4"
    loads = "".join(f"  %s{i} = getelementptr {ty}, {ptr} %src, i64 {i}{at}\n"
                    f"  %v{i} = load {ty}, {ptr} %s{i}, align 4{at}\n" for i in range(6))
    stores = "".join(f"  %d{i} = getelementptr {ty}, {ptr} %dst, i64 {i}{at}\n"
                     f"  store {ty} %v{i}, {ptr} %d{i}, align 4{at}\n" for i in reversed(range(6)))
    return (f"define void @witness({ptr} %src, {ptr} %dst) !dbg !3 {{\n{loads}{stores}  ret void{at}\n}}\n"
            "!llvm.dbg.cu = !{!0}\n!llvm.module.flags = !{!5}\n"
            "!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: LineTablesOnly)\n"
            f"!1 = !DIFile(filename: \"{loop_file}\", directory: \"\")\n"
            "!2 = !DISubroutineType(types: !{})\n"
            "!3 = distinct !DISubprogram(name: \"witness\", scope: !1, file: !1, line: 1, type: !2, "
            "spFlags: DISPFlagDefinition, unit: !0)\n"
            "!4 = !DILocation(line: 1, column: 0, scope: !3)\n"
            "!5 = !{i32 2, !\"Debug Info Version\", i32 3}\n")


def _check_llc_counts_vector_spills(target, space):
    """What build.read_places decides on: the remark's NumVectorSpills/NumVectorReloads, in total and by
    location, and VectorSpilledRegisterUnits. An llc that never writes them would read as no spill, or
    a place sized by nothing, so a spill it must make in loop 1 has to show in every one."""
    from .lowering.elf.build import read_places
    from .lowering.passes.p06_legalize import LOOP_FILE
    with tempfile.TemporaryDirectory() as d:
        remarks = os.path.join(d, "witness.regalloc.yaml")
        argv = [LLC, f"-march={target.llc_march()}", f"-mattr={target.llc_mattr()}", "-O3",
                f"--riscv-v-vector-bits-max={target.vlen_bits}", f"-pass-remarks-output={remarks}",
                "-pass-remarks-filter=regalloc", "-o", os.devnull, "-"]
        got = subprocess.run(argv, input=_vector_spill_ir(space, target.vlen_bits // 4, LOOP_FILE),
                             capture_output=True, text=True, timeout=120)
        counted = read_places(remarks) if got.returncode == 0 else None
    if (counted is None or not (counted.spills and counted.reloads) or not counted.loops.get(1)
            or not counted.units.get(("loop", 1))):
        raise ToolchainError(
            f"{LLC} does not count vector spills in its regalloc remark: a function that must spill "
            f"vectors in loop 1 exited {got.returncode} with {counted} (NumVectorSpills, "
            f"NumVectorReloads, by location, VectorSpilledRegisterUnits)"
            f"\n$ {' '.join(argv)}\n{got.stderr}")


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
    span = f"0x{machine.SPAD_VADDR:x}-0x{machine.SPAD_VADDR + machine.banked_bytes():x}"
    print(f"{'SPAD_MAP':<20} {'ok':<8} banked window {span} "
          f"({machine.VECTORLANE_SIZE} lanes x {machine.SPAD_SIZE} B)")

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


_TOGSIM_VERSION = []


def togsim_version():
    """(text, number) of the version TOGSIM_INCLUDE/togsim_runtime.h states -- TOGSIM_VERSION_STRING
    and TOGSIM_VERSION, read by compiling and running a program that prints them -- or None when the
    header is missing or states neither."""
    if _TOGSIM_VERSION:
        return _TOGSIM_VERSION[0]
    src = ('#include "togsim_runtime.h"\n#include <cstdio>\n'
           'int main() { std::printf("%s %d\\n", TOGSIM_VERSION_STRING, (int)TOGSIM_VERSION); }\n')
    got = None
    with tempfile.TemporaryDirectory() as d:
        cpp, exe = os.path.join(d, "v.cpp"), os.path.join(d, "v")
        with open(cpp, "w") as fh:
            fh.write(src)
        built = subprocess.run([HOST_CXX, "-DTOGSIM_HOST", "-I", TOGSIM_INCLUDE, cpp, "-o", exe],
                               capture_output=True, text=True)
        if built.returncode == 0:
            ran = subprocess.run([exe], capture_output=True, text=True)
            text, _, number = ran.stdout.strip().partition(" ")
            got = (text, int(number)) if ran.returncode == 0 and number.isdigit() else None
    _TOGSIM_VERSION.append(got)
    return got


def _check_togsim_header():
    path = os.path.join(TOGSIM_INCLUDE, "togsim.hpp")
    cxx = _which(HOST_CXX)
    ver = togsim_version()
    if ver is None:
        print(f"{'TOGSIM_HEADER':<20} {'MISSING':<8} {path} (or togsim_runtime.h's TOGSIM_VERSION)")
        print(f"{'':<29} only --tog needs it (tog.so is compiled against it)")
    else:
        print(f"{'TOGSIM_HEADER':<20} {'ok':<8} {path}")
        print(f"{'':<29} TOGSim v{ver[0]}")
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
