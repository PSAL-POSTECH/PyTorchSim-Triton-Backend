"""Stage `triton-compile` -- a kernel file's SPEC to the tensor IR stage 3 reads, in one process on an
unmodified Triton with the torchsim extension plugged in: the Triton frontend (01), the no-ops nothing
below can parse taken out of the text (02), TTGIR with every layout decided (02a, 02b, 02h), the
lowering to linalg/memref with every encoding carried down (03a), the layouts regularized (03)."""
import os
import re
import sys


TTIR_OPTIONS = {
    "supported_fp8_dtypes": ("fp8e5", "fp8e4b8", "fp8e4b15", "fp8e5b16"),
}

TTGIR_TARGET = "torchsim:0"

TO_LINALG = ("triton-to-structured{split-stepped-axes=true},cse,canonicalize,remove-dead-values,"
             "canonicalize,triton-to-unstructured,"
             "triton-arith-to-linalg{tensor-ptr-to-linalg=true transpose-reduce-to-rank0=false},"
             "triton-ptr-to-memref,triton-to-ptr,canonicalize,cse")

NOOPS = ("llvm.intr.assume", "ttg.barrier")
_NOOP_ALT = "|".join(re.escape(n) for n in NOOPS)
_NOOP_LINES = re.compile(r'^[ \t]*(?:' + _NOOP_ALT + r')\b[^\n]*\n?', re.M)
_NOOP_LEFT = re.compile(r'^[^\n]*\b(?:' + _NOOP_ALT + r')\b[^\n]*', re.M)


def drop_noops(text):
    out, n = _NOOP_LINES.subn("", text)
    left = _NOOP_LEFT.search(out)
    if left:
        sys.exit("a no-op the text rule cannot take: %r -- nothing below can parse it either"
                 % left.group(0).strip())
    return out, n


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
    return str(module)


def emit_linalg(clean, files, lanes):
    """02-ttclean.mlir through 02a, 02b, 02h, 03a and 03."""
    from triton._C.libtriton import ir, passes

    ctx = ir.context()
    ir.load_dialects(ctx)

    def step(src, out, build):
        mod = ir.parse_mlir_module(src, ctx)
        pm = ir.pass_manager(ctx)
        build(pm)
        passes.plugin.write_generic(pm, [out])
        pm.run(mod, out)

    convert = ("torchsim-convert-triton-to-tritongpu{target=%s num-warps=1 threads-per-warp=%d num-ctas=1}"
               % (TTGIR_TARGET, lanes))
    step(clean, files["02a"], lambda pm: passes.plugin.pipeline(pm, [convert]))
    step(files["02a"], files["02b"], lambda pm: passes.plugin.anchor_layouts(pm))
    step(files["02b"], files["02h"], lambda pm: (passes.plugin.remove_layout_conversions(pm),
                                                passes.plugin.strip_pins(pm),
                                                passes.common.add_canonicalizer(pm),
                                                passes.common.add_cse(pm)))
    step(files["02h"], files["03a"], lambda pm: passes.plugin.pipeline(pm, [TO_LINALG]))
    step(files["03a"], files["03"], lambda pm: passes.plugin.pipeline(pm, ["tritongpu-torchsim-regularize-layouts"]))


def backend_available():
    """(ok, message) -- whether this interpreter's triton can target `cpu`."""
    try:
        import triton
    except Exception as e:  # noqa: BLE001
        return False, f"cannot import triton ({type(e).__name__}: {e})"
    have = sorted(triton.backends.backends)
    if "torchsim" not in have:
        return False, (f"triton {triton.__version__} at {triton.__file__} "
                       f"offers {have} -- no `torchsim`, so the frontend cannot target "
                       f"`cpu`. That backend is the torchsim extension's, linked into "
                       f"triton/backends/torchsim.")
    return True, f"triton {triton.__version__} ({', '.join(have)})"


def _main(argv):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from pytorchsim_triton_compiler.contract.kernel_spec import load_spec

    kernel, lanes = argv[1], int(argv[2])
    files = dict(zip(("01", "02", "02a", "02b", "02h", "03a", "03"), argv[3:10]))

    print("[triton-compile] frontend", flush=True)
    ttir = emit_ttir(load_spec(kernel))
    with open(files["01"], "w") as f:
        f.write(ttir)

    clean, n = drop_noops(ttir)
    if n:
        print(f"[triton-compile] {n} unparseable no-op(s) taken out of the text", flush=True)
    with open(files["02"], "w") as f:
        f.write(clean)

    print("[triton-compile] TTGIR and the lowering to linalg", flush=True)
    try:
        emit_linalg(files["02"], files, lanes)
    except RuntimeError:
        sys.exit(1)


if __name__ == "__main__":
    _main(sys.argv)
