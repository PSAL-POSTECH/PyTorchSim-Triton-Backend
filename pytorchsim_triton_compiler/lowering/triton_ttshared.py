"""Stage `triton-shared` -- TTIR to the tensor IR stage 3 reads, on an unmodified Triton with the
torchsim extension plugged in: TTGIR with every layout decided, triton_shared's lowering, the
layouts regularized. Each step's module is written as a stage file.

Run as a script this is the stage-2 worker; see `_main`.
"""
import sys

# triton_shared's stage as one pipeline string over the extension's registered passes.
TTSHARED = ("triton-to-structured{split-stepped-axes=true},cse,canonicalize,remove-dead-values,"
            "canonicalize,triton-to-unstructured,"
            "triton-arith-to-linalg{tensor-ptr-to-linalg=true transpose-reduce-to-rank0=false},"
            "triton-ptr-to-memref,triton-to-ptr,canonicalize,cse")


def emit(clean, files, target, lanes):
    """02-ttclean.mlir through 02a, 02b, 02h, 03a and 03. `files` maps each step to its path; each
    step reads the file the one before wrote, as the opt tool this replaces did."""
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
               % (target, lanes))
    step(clean, files["02a"], lambda pm: passes.plugin.pipeline(pm, [convert]))
    step(files["02a"], files["02b"], lambda pm: passes.plugin.anchor_layouts(pm))
    step(files["02b"], files["02h"], lambda pm: (passes.plugin.remove_layout_conversions(pm),
                                                passes.plugin.strip_pins(pm),
                                                passes.common.add_canonicalizer(pm),
                                                passes.common.add_cse(pm)))
    step(files["02h"], files["03a"], lambda pm: passes.plugin.pipeline(pm, [TTSHARED]))
    step(files["03a"], files["03"], lambda pm: passes.plugin.pipeline(pm, ["tritongpu-torchsim-regularize-layouts"]))


def _main(argv):
    """The stage-2 worker, in its OWN process: libtriton.so carries an MLIR of its own, and beside
    the bindings' one the dialect registries cross. Only the written .mlir files cross back.
    argv: clean target lanes 02a 02b 02h 03a 03."""
    clean, target, lanes = argv[1], argv[2], int(argv[3])
    files = dict(zip(("02a", "02b", "02h", "03a", "03"), argv[4:9]))
    try:
        emit(clean, files, target, lanes)
    except RuntimeError:
        # The diagnostics are on stderr already, as the opt tool wrote them.
        sys.exit(1)


if __name__ == "__main__":
    _main(sys.argv)
