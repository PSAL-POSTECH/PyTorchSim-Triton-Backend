
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.runtime.errors import OutOfResources
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import base64
import hashlib
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

@functools.lru_cache()
def _pipeline_fingerprint():
    """sha256 over the source of every .py in PTB's package: Triton's cache key holds it, so a binary
    another pipeline built is never served."""
    import pytorchsim_triton_compiler as package
    root = os.path.dirname(os.path.abspath(package.__file__))
    digest = hashlib.sha256()
    for here, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(f for f in files if f.endswith(".py")):
            path = os.path.join(here, name)
            digest.update(os.path.relpath(path, root).encode())
            with open(path, "rb") as f:
                digest.update(f.read())
    return digest.hexdigest()



COLLECT = "PYTORCHSIM_COLLECT_FAILURES"
"""A directory: a compile that fails (not on capacity) is moved there with its reason and answered with a
binary that launches nothing, so one run of a model lists every kernel that fails. A debugging switch."""


def _collect(work, spec, ttir, stopped, done, metadata):
    """Move the failed compile's workdir under COLLECT, record why it failed, and yield the unlaunchable binary."""
    name = (re.findall(r"tt\.func \w+ @(\w+)", ttir) or ["kernel"])[0]
    dest = os.path.join(os.environ[COLLECT], f"{name}-{os.path.basename(work)[-8:]}")
    shutil.move(work, dest)
    with open(os.path.join(dest, "collected.json"), "w") as f:
        json.dump({"name": name, "reason": (stopped or {}).get("reason"), "kind": (stopped or {}).get("kind"),
                   "message": (stopped or {}).get("message") or (done.stdout + done.stderr)[-1500:]}, f, indent=1)
    metadata["name"], metadata["kernel_object"], metadata["artifacts"] = name, "", "{}"
    return ttir.encode()


def build(ttir, spec_text, tog, target, keep, metadata):
    """The ELF of `ttir` under `spec_text` from PTB's pipeline in a process of its own; its kernel object and
    artifacts go to `metadata`. Only capacity drops a config: a scratchpad or register overflow is OutOfResources.
    A named refusal is a compiler gap and, like any other stop, a RuntimeError: autotune must not hide it."""
    from pytorchsim_triton_compiler.contract import failure
    spec = json.loads(spec_text)
    work = tempfile.mkdtemp(prefix="pytorchsim-ptb-")
    with open(os.path.join(work, "spec.ttir"), "w") as f:
        f.write(ttir)
    with open(os.path.join(work, "spec.json"), "w") as f:
        f.write(spec_text)
    env = dict(os.environ)
    if target:
        env["PYTORCHSIM_COMPILE_TARGET"] = target
    done = subprocess.run([sys.executable, "-m", "pytorchsim_triton_compiler.cli",
                           os.path.join(work, "spec.json"), "--workdir", work]
                          + (["--tog"] if tog else []),
                          capture_output=True, text=True, env=env)
    stopped = failure.read(work) if done.returncode else None
    if stopped and stopped["kind"] == failure.SPAD_OVERFLOW:
        shutil.rmtree(work, ignore_errors=True)
        raise OutOfResources(stopped["usage"], stopped["budget"], "scratchpad")
    if stopped and stopped["kind"] == failure.SPILL_NOT_ALLOWED:
        shutil.rmtree(work, ignore_errors=True)
        raise OutOfResources("a vector register spill", "none allowed", "vector registers")
    if done.returncode and os.environ.get(COLLECT):
        return _collect(work, spec, ttir, stopped, done, metadata)
    if stopped and stopped["kind"] == failure.REFUSED:
        raise RuntimeError(f"pytorchsim-triton-compiler cannot lower {spec.get('name', 'a kernel')} yet, a compiler "
                           f"gap ({stopped['reason']}), kept in {work}: {stopped['message']}")
    if done.returncode:
        raise RuntimeError(f"pytorchsim-triton-compiler failed on {spec.get('name', 'a kernel')} (kept in {work}):\n"
                           f"{done.stdout[-2000:]}{done.stderr[-2000:]}")
    with open(os.path.join(work, "kernel.json")) as f:
        metadata["kernel_object"] = f.read()
    metadata["name"] = json.loads(metadata["kernel_object"])["name"]
    kinds = json.loads(metadata["kernel_object"])["artifacts"]
    wanted = ("machine_config",) + (("tog_so", "tog_cpp", "tile_types", "cycle_elf") if tog else ())
    files = {}
    for kind in wanted:
        with open(os.path.join(work, kinds[kind]), "rb") as f:
            files[kind] = {"name": kinds[kind], "data": base64.b64encode(f.read()).decode()}
    for node in json.loads(metadata["kernel_object"]).get("copy_nodes", ()) if tog else ():
        with open(os.path.join(work, node["tog_so"]), "rb") as f:
            files[node["tog_so"]] = {"name": node["tog_so"], "data": base64.b64encode(f.read()).decode()}
    metadata["artifacts"] = json.dumps(files)
    with open(os.path.join(work, kinds["elf"]), "rb") as f:
        elf = f.read()
    if keep:
        metadata["workdir"] = work
    else:
        shutil.rmtree(work, ignore_errors=True)
    return elf



@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True
    pytorchsim_spec: str = ""
    """JSON, contract.kernel_spec.JSON_FIELDS: what the pipeline reads of the kernel that its Triton IR does
    not hold. Empty: the backend stops at TTIR. It is part of the hash, so a retiled kernel is another."""

    pytorchsim_tog: bool = False
    """Timing mode: the stage also builds the TOG (--tog), its files joining metadata["artifacts"]."""
    pytorchsim_keep_workdir: bool = False
    """Keep the stage's workdir for its dumps (also PYTORCHSIM_KEEP_WORKDIR=1); metadata["workdir"] names it."""
    pytorchsim_target: str = ""
    """The machine, a name under $TOGSIM_DIR/configs or a path; empty: PYTORCHSIM_COMPILE_TARGET."""

    def __post_init__(self):
        pass

    def hash(self):
        fields = dict(self.__dict__, pytorchsim_target=self.pytorchsim_target
                      or os.environ.get("PYTORCHSIM_COMPILE_TARGET", ""))
        key = '_'.join([f'{name}-{val}' for name, val in fields.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = 'obj'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend in ('cpu', 'npu')

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns


    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_obj(mod, metadata, options):
        """The TTIR through PTB's pipeline, built for the spec's extents. Returns the ELF; the kernel
        object goes to metadata with what rebuild_for needs to build the same kernel for other extents."""
        ttir = re.sub(r" \{tt\.divisibility = \d+ : i32\}", "", str(mod))
        target = options.pytorchsim_target or os.environ.get("PYTORCHSIM_COMPILE_TARGET", "")
        keep = options.pytorchsim_keep_workdir or os.environ.get("PYTORCHSIM_KEEP_WORKDIR") == "1"
        elf = build(ttir, options.pytorchsim_spec, options.pytorchsim_tog, target, keep, metadata)
        metadata["rebuild"] = json.dumps({"ttir": ttir, "spec": options.pytorchsim_spec,
                                          "tog": options.pytorchsim_tog, "target": target})
        return elf

    @staticmethod
    def rebuild_for(rebuild, spans, scalars):
        """(ELF, kernel object, artifacts) of the kernel `rebuild` (a binary's metadata) for other extents: each
        tensor's storage span and the values of the scalars its spec states."""
        spec = json.loads(rebuild["spec"])
        for arg, span in zip(spec["args"], spans):
            arg["shape"] = [span]
        spec["extra"]["scalar_values"] = {**spec["extra"].get("scalar_values", {}), **scalars}
        got = {}
        elf = build(rebuild["ttir"], json.dumps(spec), rebuild["tog"], rebuild["target"], False, got)
        return elf, got["kernel_object"], got["artifacts"]

    @staticmethod
    def make_ttir_only(mod, metadata):
        """What a compile without pytorchsim_spec yields: the TTIR as the binary, with no kernel object, so
        the kernel is accepted by Triton and cannot be launched."""
        metadata["name"] = mod.get_entry_func_name()
        metadata["kernel_object"] = ""
        metadata["artifacts"] = "{}"
        return str(mod).encode()

    def pack_metadata(self, metadata):
        return ()

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        if options.pytorchsim_spec:
            stages["obj"] = lambda src, metadata: self.make_obj(src, metadata, options)
        else:
            stages["obj"] = lambda src, metadata: self.make_ttir_only(src, metadata)


    @functools.lru_cache()
    def hash(self):
        return f"{self.target}-{_pipeline_fingerprint()}"

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}
