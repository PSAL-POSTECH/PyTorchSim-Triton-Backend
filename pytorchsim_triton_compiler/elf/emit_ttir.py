"""Stage `triton-compile` -- the Triton frontend: a kernel file's SPEC to Triton IR.

Run as a script this is the stage-1 worker; see `_main`.
"""
import os
import sys


TTIR_OPTIONS = {
    "supported_fp8_dtypes": ("fp8e5", "fp8e4b8", "fp8e4b15", "fp8e5b16"),
}


def _make_ttir(mod, opts):
    """Triton's own ttir pass pipeline, run in place."""
    from triton._C.libtriton import ir, passes

    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
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


def emit(spec, out_path):
    """Write `spec`'s Triton IR to `out_path`. Returns it."""
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

    _make_ttir(module, opts)

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
    """The stage-1 worker, in its OWN process: triton's libtriton carries an MLIR
    of its own, and beside the bindings' one the dialect registries cross. The
    flag keeps this process off MLIR; only the written .mlir crosses back."""
    os.environ["TORCHSIM_COMPILE_TTIR_ONLY"] = "1"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from pytorchsim_triton_compiler.contract.kernel_spec import load_spec

    emit(load_spec(argv[1]), argv[2])


if __name__ == "__main__":
    _main(sys.argv)
