"""Build the gem5 cycle-sampling binary out of a marker-annotated kernel."""
import os
import sys

from ... import machine
from ...contract import kernel_object
from ...metrics import wallclock
from ...stage import Ctx
from ..elf.build import build_elf

SAMPLE_MLIR = "04-sample.mlir"
CYCLE_BIN = "cycle_bin"

FILENAME = "timing-cycle.json"


def build_cycle_binary(spec, kernel_path, workdir, sample=SAMPLE_MLIR,
                       out=CYCLE_BIN, grouping="typed"):
    """Lower the marker-annotated sample IR to a gem5-runnable binary."""
    from ..passes import run_standard_lowering

    from .passes.build_tog import run_tog

    ctx = Ctx(spec, kernel_path, workdir)
    src = ctx.path(sample)

    with wallclock.PROF.step("sample-ir"):
        run_tog(kernel_object.artifact(workdir, "post_vcix_ir"),
                ctx.path("tog_sample.py"), src, sample_mode=True, grouping=grouping)

    lowered = ctx.path("04-sample-lowered.mlir")
    run_standard_lowering(src, lowered, timing=True,
                          vectorlane=machine.VECTORLANE_SIZE)
    elf, _ = build_elf(
        ctx, "cycle", "cycle",
        lowered=lowered,
        ll=ctx.path("05-cycle.ll"),
        obj=ctx.path("05-cycle.o"),
        asm=ctx.path("05-cycle.s"),
        wrap_c=ctx.path("05-cycle-wrapper.c"),
        elf=ctx.path(out),
        single_program=True)
    kernel_object.add_artifact(ctx.workdir, "cycle_elf", os.path.basename(elf))
    return elf


def main(argv):
    """Command line: build the cycle binary for one spec, and time the build."""
    import argparse

    from ...contract.kernel_spec import load_spec

    ap = argparse.ArgumentParser(prog="pytorchsim_triton_compiler.lowering.tog.cycle_binary", description=__doc__)
    ap.add_argument("spec", help="kernel spec .py")
    ap.add_argument("workdir")
    ap.add_argument("--sample", default=SAMPLE_MLIR)
    ap.add_argument("--out", default=CYCLE_BIN)
    args = ap.parse_args(argv[1:])

    spec = load_spec(args.spec)
    wallclock.PROF.label = spec.name
    try:
        with wallclock.PROF.stage("cycle-binary"):
            print(build_cycle_binary(spec, args.spec, args.workdir,
                                     sample=args.sample, out=args.out))
    finally:
        wallclock.PROF.write(os.path.join(os.path.abspath(args.workdir), FILENAME))
        if wallclock.want_report():
            print(wallclock.PROF.report())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
