"""Stage `triton-shared` -- TTIR to the tensor IR stage 3 reads, on an unmodified Triton with the
torchsim extension plugged in: TTGIR with every layout decided, triton_shared's lowering, the
layouts regularized."""
import sys

TTSHARED = ("triton-to-structured{split-stepped-axes=true},cse,canonicalize,remove-dead-values,"
            "canonicalize,triton-to-unstructured,"
            "triton-arith-to-linalg{tensor-ptr-to-linalg=true transpose-reduce-to-rank0=false},"
            "triton-ptr-to-memref,triton-to-ptr,canonicalize,cse")


def emit(clean, files, target, lanes):
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
    clean, target, lanes = argv[1], argv[2], int(argv[3])
    files = dict(zip(("02a", "02b", "02h", "03a", "03"), argv[4:9]))
    try:
        emit(clean, files, target, lanes)
    except RuntimeError:
        sys.exit(1)


if __name__ == "__main__":
    _main(sys.argv)
