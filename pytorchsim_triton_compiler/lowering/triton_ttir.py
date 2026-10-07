"""Stage `triton-compile` -- the Triton frontend: a kernel file's SPEC to Triton IR."""
import os
import sys


TTIR_OPTIONS = {
    "supported_fp8_dtypes": ("fp8e5", "fp8e4b8", "fp8e4b15", "fp8e5b16"),
}


def emit(spec, out_path):
    """Write `spec`'s Triton IR to `out_path`."""
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

    with open(out_path, "w") as f:
        f.write(str(module))
    return out_path


def backend_available():
    """(ok, message) -- whether this interpreter's triton can target `cpu`."""
    try:
        import triton
    except Exception as e:  # noqa: BLE001
        return False, f"cannot import triton ({type(e).__name__}: {e})"
    have = sorted(triton.backends.backends)
    if "torchsim" not in have:
        return False, (f"triton {triton.__version__} at {triton.__file__} "
                       f"offers {have} -- no `torchsim`, so stage 1 cannot target "
                       f"`cpu`. That backend is the torchsim extension's, linked into "
                       f"triton/backends/torchsim.")
    return True, f"triton {triton.__version__} ({', '.join(have)})"


def _main(argv):
    os.environ["TORCHSIM_COMPILE_TTIR_ONLY"] = "1"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from pytorchsim_triton_compiler.contract.kernel_spec import load_spec

    emit(load_spec(argv[1]), argv[2])


if __name__ == "__main__":
    _main(sys.argv)
