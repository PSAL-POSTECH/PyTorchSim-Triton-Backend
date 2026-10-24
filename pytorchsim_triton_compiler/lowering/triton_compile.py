"""Stage triton-compile: a kernel file's SPEC -> 07-regularized.mlir, the tensor IR torchsim-compile
reads, on Triton with the torchsim extension plugged in (setup/triton.sh builds it).

  01-ttir         the Triton frontend, where an atomic is refused
  02-ttclean      the no-ops nothing below can parse erased
  03..07          STEPS: TTGIR with every layout decided, lowered to linalg/memref with every
                  encoding carried down, the layouts regularized

run(ctx) is the stage. It runs this same file as a worker in a process of its own -- libtriton
carries an MLIR of its own, and beside the bindings' one the dialect registries cross -- and then
checks, on the bindings' side, that 06-linalg kept a layout on every tile."""
import os
import sys

ARTIFACTS = ("01-ttir.mlir", "02-ttclean.mlir", "03-ttgir.mlir", "04-anchored.mlir",
             "05-layouts.mlir", "06-linalg.mlir", "07-regularized.mlir")

TTIR_OPTIONS = {
    "supported_fp8_dtypes": ("fp8e5", "fp8e4nv", "fp8e4b8", "fp8e4b15", "fp8e5b16"),
}

TTGIR_TARGET = "torchsim:0"

TO_LINALG = ("triton-to-structured{split-stepped-axes=true},cse,canonicalize,remove-dead-values,"
             "canonicalize,triton-to-unstructured,"
             "triton-arith-to-linalg{tensor-ptr-to-linalg=true transpose-reduce-to-rank0=false},"
             "triton-ptr-to-memref,triton-to-ptr,canonicalize,cse")

NOOPS = ("llvm.intr.assume", "ttg.barrier")

ATOMICS = ("tt.atomic_rmw", "tt.atomic_cas")


def drop_noops(module, ctx):
    """Erase every NOOPS op of `module` in place, through the plugin's pass; how many went."""
    from triton._C.libtriton import ir, passes
    found = []
    module.walk(lambda op: found.append(op.get_name()) if op.get_name() in NOOPS else None)
    if found:
        pm = ir.pass_manager(ctx)
        passes.plugin.drop_ops(pm, list(NOOPS))
        pm.run(module, "drop_noops")
    return len(found)


def refuse_atomics(module):
    """Refuse `module` if it holds an atomic: they are refused by decision, before any lowering
    meets one (a tt.descriptor_reduce is a tt.atomic_rmw by now, make_ttir rewrote it)."""
    from pytorchsim_triton_compiler.contract.stop import Reason, Refused
    found = set()
    module.walk(lambda op: found.add(op.get_name()) if op.get_name() in ATOMICS else None)
    if found:
        raise Refused(Reason.ATOMICS,
                      f"{', '.join(sorted(found))} in 01-ttir.mlir: atomics are refused on this "
                      f"backend, and that is a decision rather than a lowering still missing")


def op_counts(path, ctx):
    """Op name -> count over the module at `path`, parsed by this Triton."""
    from collections import Counter
    from triton._C.libtriton import ir
    counts = Counter()
    ir.parse_mlir_module(path, ctx).walk(lambda op: counts.update([op.get_name()]))
    return dict(counts)


def emit_ttir(spec):
    """`spec`'s Triton IR, as text."""
    from triton.compiler.compiler import ASTSource, make_backend
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import ir

    fn = spec.kernel()

    src = ASTSource(fn=fn, signature=spec.signature, constexprs=spec.constexprs)

    target = GPUTarget("cpu", 0, 0)
    backend = make_backend(target)

    ttir_options = dict(TTIR_OPTIONS)
    ttir_options.update(spec.extra.get("ttir_options", {}))
    opts = backend.parse_options(ttir_options)

    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)

    module = src.make_ir(target, opts,
                         backend.get_codegen_implementation(opts),
                         backend.get_module_map(),
                         ctx)

    backend.make_ttir(module, {}, opts)
    return module, ctx


def _convert(pm, passes, lanes):
    passes.plugin.pipeline(pm, ["torchsim-convert-triton-to-tritongpu{target=%s num-warps=1 "
                                "threads-per-warp=%d num-ctas=1}" % (TTGIR_TARGET, lanes)])


def _anchor(pm, passes, lanes):
    from pytorchsim_triton_compiler import machine
    passes.plugin.pipeline(pm, ["tritongpu-torchsim-anchor-layouts{group-bits=%d}"
                                % (machine.VLEN_BITS * machine.ISA_MAX_LMUL)])


def _propagate(pm, passes, lanes):
    passes.plugin.remove_layout_conversions(pm)
    passes.plugin.strip_pins(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)


def _to_linalg(pm, passes, lanes):
    passes.plugin.pipeline(pm, [TO_LINALG])


def _regularize(pm, passes, lanes):
    passes.plugin.pipeline(pm, ["tritongpu-torchsim-regularize-layouts"])


STEPS = [
    ("03-ttgir.mlir", _convert, "torchsim-convert-triton-to-tritongpu"),
    ("04-anchored.mlir", _anchor, "tritongpu-torchsim-anchor-layouts"),
    ("05-layouts.mlir", _propagate, "remove-layout-conversions, strip-pins, canonicalize, cse"),
    ("06-linalg.mlir", _to_linalg, "triton-to-structured .. triton-to-ptr (TO_LINALG)"),
    ("07-regularized.mlir", _regularize, "tritongpu-torchsim-regularize-layouts"),
]


def recipe():
    return ["the Triton frontend -> 01-ttir.mlir", "no-ops dropped -> 02-ttclean.mlir"] + [
        f"{what} -> {name}" for name, _, what in STEPS]


def run(ctx):
    from .. import machine, toolchain
    from ..contract import stop
    from ..stage import StageError, TritonPassFailed, WorkerRefused, run_tool
    stop.clear(ctx.workdir)
    try:
        run_tool(ctx, "triton-compile",
                 [sys.executable, os.path.abspath(__file__), ctx.kernel_path,
                  machine.VECTORLANE_SIZE, ctx.workdir], env=toolchain.triton_env())
    except StageError as e:
        said = stop.recorded(ctx.workdir)
        if said is None:
            raise TritonPassFailed(e.stage, str(e), cmd=e.cmd, stderr=e.stderr) from e
        raise WorkerRefused(said[0], e.stage, said[1], cmd=e.cmd, stderr=e.stderr) from e
    _check_layouts_carried(ctx.path("06-linalg.mlir"))
    return ctx.path("07-regularized.mlir")


def _check_layouts_carried(path):
    from .. import toolchain
    from ..stage import StageError
    from .passes.lib_ir import is_reshape_shape, is_type, name_of, op_of, walk
    toolchain.ensure_bindings()
    from mlir.ir import Context, Location, Module, RankedTensorType
    with open(path) as f:
        text = f.read()
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    bare = []
    with ctx, Location.unknown():
        module = Module.parse(text)
        for raw in walk(module.operation):
            op = op_of(raw)
            values = list(op.results) + [a for r in raw.regions for b in r.blocks for a in b.arguments]
            for v in values:
                if not isinstance(v.type, RankedTensorType):
                    continue
                ty = RankedTensorType(v.type)
                elem = ty.element_type
                if (ty.rank == 0 or ty.encoding is not None or is_type(elem, "tt", "ptr")
                        or is_reshape_shape(v)):
                    continue
                bare.append("%s : %s" % (name_of(op), ty))
    if bare:
        raise StageError("triton-compile", "%s: %d tile(s) without a layout encoding, first %s"
                         % (os.path.basename(path), len(bare), "; ".join(bare[:3])),
                         artifact=path)


def emit_linalg(clean, workdir, lanes):
    """02-ttclean.mlir through STEPS, each read from the one before and written to its artifact."""
    from triton._C.libtriton import ir, passes

    ctx = ir.context()
    ir.load_dialects(ctx)
    src = clean
    for name, build, _ in STEPS:
        out = os.path.join(workdir, name)
        mod = ir.parse_mlir_module(src, ctx)
        pm = ir.pass_manager(ctx)
        build(pm, passes, lanes)
        passes.plugin.write_generic(pm, [out])
        pm.run(mod, out)
        src = out


def backend_available():
    """(ok, message) -- whether this interpreter's triton can target `cpu`."""
    try:
        import triton
    except Exception as e:  # noqa: BLE001
        return False, f"cannot import triton ({type(e).__name__}: {e})"
    have = sorted(triton.backends.backends)
    if "pytorchsim" not in have:
        return False, (f"triton {triton.__version__} at {triton.__file__} "
                       f"offers {have} -- no `pytorchsim`, so the frontend cannot target "
                       f"`cpu`. That backend is the torchsim extension's, linked into "
                       f"triton/backends/pytorchsim.")
    return True, f"triton {triton.__version__} ({', '.join(have)})"


def _main(argv):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from pytorchsim_triton_compiler.contract import stop
    from pytorchsim_triton_compiler.contract.kernel_spec import load_spec

    if argv[1] == "--count-ops":
        import json
        from triton._C.libtriton import ir
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import make_backend
        ctx = ir.context()
        ir.load_dialects(ctx)
        make_backend(GPUTarget("cpu", 0, 0)).load_dialects(ctx)
        print(json.dumps({path: op_counts(path, ctx) for path in argv[2:]}))
        return

    kernel, lanes, workdir = argv[1], int(argv[2]), argv[3]
    ttir_path, clean_path = (os.path.join(workdir, a) for a in ARTIFACTS[:2])

    print("[triton-compile] frontend", flush=True)
    module, ctx = emit_ttir(load_spec(kernel))
    with open(ttir_path, "w") as f:
        f.write(str(module))
    try:
        refuse_atomics(module)
    except stop.Refusal as e:
        stop.record(workdir, e)
        raise

    n = drop_noops(module, ctx)
    if n:
        print(f"[triton-compile] {n} no-op(s) erased before the lowering", flush=True)
    with open(clean_path, "w") as f:
        f.write(str(module))

    print("[triton-compile] TTGIR and the lowering to linalg", flush=True)
    try:
        emit_linalg(clean_path, workdir, lanes)
    except RuntimeError:
        sys.exit(1)


if __name__ == "__main__":
    _main(sys.argv)
