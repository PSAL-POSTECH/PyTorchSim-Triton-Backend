"""The Triton backend plugin as a caller reaches it: triton.compile with pytorchsim_spec runs the pipeline and
yields the ELF, a scratchpad overflow is OutOfResources, a launch goes to the registered launcher or raises.
Each case runs in a Triton process (toolchain.triton_env), as the MLIR bindings cannot share one with it.
    python3 tests/runtime/triton_backend_cases.py        exit 0 iff every case holds"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("PYTORCHSIM_COMPILE_TARGET", suite_target.NAME)
SPAD = "tests/cases/mechanism/tile/tile_spad_over_budget.py"
ADD = "tests/cases/smoke/add.py"


def _compile(path, device_type="cpu", derive=False, with_spec=True, attrs=None, constexprs=None, **options):
    """triton.compile of the kernel file `path` through the backend, with its spec as pytorchsim_spec."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    from pytorchsim_triton_compiler.contract.kernel_spec import load_spec
    spec = load_spec(os.path.join(ROOT, path))
    text = json.dumps({f: ([vars(a) for a in spec.args] if f == "args" else getattr(spec, f))
                       for f in ("name", "signature", "args", "grid", "grid_rule", "graph_prefetch",
                                 "parallel_axes", "extra")})
    if derive:
        got = json.loads(text)
        got.pop("name"), got.pop("signature")
        got["args"] = [{"role": a["role"], "shape": a["shape"]} for a in got["args"]]
        text = json.dumps(got)
    src = ASTSource(fn=spec.kernel(), signature=spec.signature, constexprs={**spec.constexprs, **(constexprs or {})}, attrs=attrs)
    return triton.compile(src, target=GPUTarget(device_type, 0, 0), options={**({"pytorchsim_spec": text} if with_spec else {}), **options})


def _worker_overflow_is_out_of_resources():
    from triton.runtime.errors import OutOfResources
    try:
        _compile(SPAD)
    except OutOfResources as e:
        return e.name == "scratchpad" and e.required > e.limit > 0, str(e)
    return False, "compiled"


LN = "tests/cases/mechanism/gather/gather_layer_norm_permuted_store_on_a_cut_dim.py"
GAP = "tests/cases/mechanism/mask/mask_loaded_mask_two_axes_before_the_tensor.py"


def _worker_compiler_gap_is_an_error_not_a_dropped_config():
    from triton.runtime.errors import OutOfResources
    try:
        _compile(GAP)
    except OutOfResources as e:
        return False, f"dropped as a capacity limit: {e}"
    except RuntimeError as e:
        return "compiler gap (masked-box-before-tensor)" in str(e), str(e)[:300]
    return False, "compiled"


def _worker_a_kernel_compiles_at_two_configs():
    got = [_compile(LN, constexprs={"XBLOCK": x}).asm["obj"][:4] for x in (64, 8192)]
    return got == [b"\x7fELF"] * 2, [str(g) for g in got]


def _worker_internal_error_stays_a_runtime_error():
    from triton.runtime.errors import OutOfResources
    try:
        _compile(ADD, pytorchsim_spec=json.dumps({"args": []}))
    except OutOfResources as e:
        return False, str(e)
    except RuntimeError as e:
        return "pytorchsim-triton-compiler failed" in str(e), str(e)[:200]
    return False, "compiled"


def _worker_small_kernel_yields_a_binary():
    got = _compile(ADD)
    elf = got.asm["obj"]
    kobj = json.loads(got.metadata.kernel_object)
    return elf[:4] == b"\x7fELF" and kobj["name"] == got.name, {"bytes": len(elf), "name": got.name}


def _worker_launch_goes_to_the_registered_launcher():
    from triton.runtime import driver
    from triton.backends.pytorchsim.driver import CPUDriver, set_launcher
    driver.set_active(CPUDriver())
    seen = []
    set_launcher(lambda binary, kobj, grid, args, artifacts=None: seen.append((bytes(binary[:4]), json.loads(kobj)["name"], grid, args)))
    _compile(ADD)[(3, 1, 1)]("a", "b", "c", 8)
    return seen == [(b"\x7fELF", "add_kernel", (3, 1, 1), ("a", "b", "c", 8))], seen


def _worker_launch_without_a_launcher_raises():
    from triton.runtime import driver
    from triton.backends.pytorchsim.driver import CPUDriver, set_launcher
    driver.set_active(CPUDriver())
    set_launcher(None)
    try:
        _compile(ADD)[(1, 1, 1)]("a", "b", "c", 8)
    except RuntimeError as e:
        return "set_launcher" in str(e), str(e)
    return False, "launched"


def _worker_backend_hash_follows_the_pipeline_source():
    import pytorchsim_triton_compiler as package
    from triton.backends.compiler import GPUTarget
    from triton.backends.pytorchsim.compiler import CPUBackend, _pipeline_fingerprint
    here = os.path.join(os.path.dirname(package.__file__), "_fingerprint_probe.py")
    before = CPUBackend(GPUTarget("cpu", 0, 0)).hash()
    try:
        with open(here, "w") as f:
            f.write("X = 1\n")
        _pipeline_fingerprint.cache_clear()
        CPUBackend.hash.cache_clear()
        after = CPUBackend(GPUTarget("cpu", 0, 0)).hash()
    finally:
        os.remove(here)
        _pipeline_fingerprint.cache_clear()
        CPUBackend.hash.cache_clear()
    return before != after, [before[-12:], after[-12:]]


def _worker_machine_comes_from_the_options():
    os.environ.pop("PYTORCHSIM_COMPILE_TARGET", None)
    got = _compile(ADD, pytorchsim_target=suite_target.NAME)
    return json.loads(got.metadata.kernel_object)["target"]["name"] == suite_target.NAME, got.name


def _worker_timing_option_returns_the_tog():
    import base64
    plain = _compile(ADD)
    timed = _compile(ADD, pytorchsim_tog=True)
    tog = json.loads(timed.metadata.artifacts)
    so = base64.b64decode(tog["tog_so"]["data"])
    return so[:4] == b"\x7fELF" and sorted(json.loads(plain.metadata.artifacts)) == ["machine_config"], sorted(tog)


def _worker_keep_option_leaves_the_workdir():
    import shutil
    gone = _compile(ADD)
    kept = _compile(ADD, pytorchsim_keep_workdir=True)
    work = kept.metadata.workdir
    ok = os.path.isdir(work) and any(n.startswith("05-") for n in os.listdir(work)) and "workdir" not in gone.metadata._asdict()
    shutil.rmtree(work, ignore_errors=True)
    return ok, work


def _worker_launch_hands_over_the_artifacts():
    from triton.runtime import driver
    from triton.backends.pytorchsim.driver import CPUDriver, set_launcher
    driver.set_active(CPUDriver())
    seen = []
    set_launcher(lambda binary, kobj, grid, args, artifacts=None: seen.append(sorted(json.loads(artifacts))))
    _compile(ADD)[(1, 1, 1)]("a", "b", "c", 8)
    _compile(ADD, pytorchsim_tog=True)[(1, 1, 1)]("a", "b", "c", 8)
    return seen == [["machine_config"], ["cycle_elf", "machine_config", "tile_types", "tog_cpp", "tog_so"]], seen


def _worker_npu_target_compiles():
    return _compile(ADD, device_type="npu").asm["obj"][:4] == b"\x7fELF", "npu"


def _worker_launch_drops_constexpr_arguments():
    from triton.runtime import driver
    from triton.backends.pytorchsim.driver import CPUDriver, set_launcher
    driver.set_active(CPUDriver())
    seen = []
    set_launcher(lambda binary, kobj, grid, args, artifacts=None: seen.append(args))
    got = _compile(ADD)
    width = len(got.src.signature)
    got[(1, 1, 1)](*[("p%d" % i) if t != "constexpr" else 128 for i, t in enumerate(got.src.signature.values())])
    return seen == [tuple(("p%d" % i) for i, t in enumerate(got.src.signature.values()) if t != "constexpr")] and width > 4, seen


def _worker_elf_is_the_one_the_object_names():
    import shutil
    got = _compile("tests/cases/mechanism/rtattr/rtattr_embedding_gather_tog.py", pytorchsim_keep_workdir=True)
    work, kobj = got.metadata.workdir, json.loads(got.metadata.kernel_object)
    with open(os.path.join(work, kobj["artifacts"]["elf"]), "rb") as f:
        named = f.read()
    others = [n for n in os.listdir(work) if n.endswith(".elf") and n != kobj["artifacts"]["elf"]]
    shutil.rmtree(work, ignore_errors=True)
    return got.asm["obj"] == named and bool(others), others


def _worker_spec_without_signature_reads_it_off_the_ir():
    full, derived = _compile(ADD), _compile(ADD, derive=True)
    a, b = json.loads(full.metadata.kernel_object), json.loads(derived.metadata.kernel_object)
    shape = lambda k: [(x["role"], x["dtype"], x["shape"]) for x in k["args"]]
    return shape(a) == shape(b) and a["name"] == b["name"] and a["scalars"] == b["scalars"] and \
        [x["name"] for x in b["args"]] == ["x_ptr", "y_ptr", "o_ptr"], [x["name"] for x in b["args"]]


def _worker_compile_without_a_spec_is_not_launchable():
    from triton.runtime import driver
    from triton.backends.pytorchsim.driver import CPUDriver, set_launcher
    driver.set_active(CPUDriver())
    set_launcher(lambda *a, **k: None)
    got = _compile(ADD, with_spec=False)
    try:
        got[(1, 1, 1)]("a", "b", "c", 8)
    except RuntimeError as e:
        return got.name == "add_kernel" and "without pytorchsim_spec" in str(e), str(e)
    return False, "launched"


def _worker_alignment_hints_are_dropped():
    import shutil
    kernel = _compile(ADD, pytorchsim_keep_workdir=True, attrs={(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]]})
    work = kernel.metadata.workdir
    with open(os.path.join(work, "spec.ttir")) as f:
        text = f.read()
    shutil.rmtree(work, ignore_errors=True)
    return "tt.divisibility" not in text, kernel.name


def _worker_launch_over_other_extents_gets_a_build_for_them():
    import torch
    from triton.runtime import driver
    from triton.backends.pytorchsim.driver import CPUDriver, set_launcher
    driver.set_active(CPUDriver())
    seen = []
    set_launcher(lambda binary, kobj, grid, args, artifacts=None: seen.append(
        [a["shape"] for a in json.loads(kobj)["args"]]))
    got = _compile(ADD)
    own, other = [torch.empty(1024)] * 3, [torch.empty(512)] * 3
    got[(1, 1, 1)](*own, 1024, 128)
    got[(1, 1, 1)](*other, 1024, 128)
    got[(1, 1, 1)](*other, 1024, 128)
    return seen == [[[1024]] * 3, [[512]] * 3, [[512]] * 3] and len(got.run.builds) == 1, seen


def _worker_timing_compile_returns_the_copy_nodes_tog():
    timed = _compile("tests/cases/mechanism/ops/ops_resident_matmul.py", pytorchsim_tog=True)
    named = [n["tog_so"] for n in json.loads(timed.metadata.kernel_object)["copy_nodes"]]
    got = json.loads(timed.metadata.artifacts)
    return bool(named) and all(n in got for n in named), [named, sorted(got)]


CASES = {name[len("_worker_"):]: fn for name, fn in globals().items() if name.startswith("_worker_")}


def _run(name):
    from pytorchsim_triton_compiler import toolchain
    with tempfile.TemporaryDirectory() as cache:
        env = dict(toolchain.triton_env(), TRITON_CACHE_DIR=cache)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", name], env=env,
                           capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1:] or [""]
    try:
        ok, got = json.loads(last[0])
    except json.JSONDecodeError:
        ok, got = False, (r.stderr or r.stdout)[-300:]
    return ok, got


def main():
    if sys.argv[1:2] == ["--worker"]:
        try:
            ok, got = CASES[sys.argv[2]]()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        print(json.dumps([bool(ok), got], default=str))
        return 0
    failed = 0
    for name in CASES:
        ok, got = _run(name)
        failed += not ok
        print(f"{name:50} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
