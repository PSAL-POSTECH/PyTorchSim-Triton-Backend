"""pytorchsim-triton-compiler -- compile a Triton kernel for the PyTorchSim NPU: the RISC-V ELF Spike
runs and, with --tog, the tile-operation-graph (TOG) TOGSim replays. Running them is the caller's -- PyTorchSim,
or tests/torchsim-compile."""
import argparse
import faulthandler
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
faulthandler.enable()
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from . import toolchain
from . import pipeline
from .contract.kernel_spec import load_spec
from .lowering.tog import DEFAULT_GROUPING, GROUPINGS


def parser():
    """The driver's arguments. tests/torchsim-compile adds its own to the same parser."""
    ap = argparse.ArgumentParser(prog="pytorchsim-triton-compiler", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel", nargs="?",
                    help="kernel file, or 'doctor' (a subcommand spelled here, so the "
                         "stage flags do not apply to it)")
    names = list(dict.fromkeys(list(pipeline.STAGES) + pipeline.steps()))
    ap.add_argument("--from", dest="from_stage", default="triton-compile", choices=names)
    ap.add_argument("--to", dest="to_stage", default=None, choices=names)
    ap.add_argument("--workdir", default=None,
                    help="where the artifacts go (default: ./work/<kernel name>)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--breakdown", action="store_true",
                    help="print where the wall clock went: stage, pass, tool")
    ap.add_argument("--tog", action="store_true",
                    help="also emit the tile-operation-graph (TOG) TOGSim replays, C++ and .so")
    ap.add_argument("--tog-grouping", choices=GROUPINGS, default=None,
                    help=f"how --tog groups compute into tiles. segment: cut at DMA, wait and "
                         f"replayed loops only. typed: by unit and opcode -- DEPRECATED, kept for "
                         f"comparison, to be removed. Default: {DEFAULT_GROUPING}")
    ap.add_argument("--list-stages", action="store_true",
                    help="print the stages and steps this build has, then exit")
    return ap


def _list_stages():
    def show(name, indent):
        print(f"{indent}{name:<17} {pipeline.STAGE_DESC[name]}")
        if name in pipeline.STAGE_MODULE:
            for line in pipeline.stage_module(name).recipe():
                print(f"{' ' * len(indent)}{'':<17}   {line}")

    for i, s in enumerate(pipeline.STAGES, 1):
        show(s, f"  {i}. ")
        for step in pipeline.STEPS_OF.get(s, []):
            show(step, "       ")
    print("  with --tog, after binary:")
    for s in pipeline.TOG_STAGES:
        show(s, "       ")


def run(ap, args):
    """Act on parsed arguments: list the stages, run doctor, or compile."""
    if args.list_stages:
        _list_stages()
        return 0

    if args.kernel is None:
        ap.print_usage(sys.stderr)
        print(f"{ap.prog}: give a kernel file, or 'doctor'", file=sys.stderr)
        return 2

    if args.kernel == "doctor":
        return 0 if toolchain.doctor() else 1

    if not os.path.exists(args.kernel):
        print(f"no such kernel file: {args.kernel}", file=sys.stderr)
        return 2

    spec = load_spec(args.kernel)
    workdir = args.workdir or os.path.join(os.getcwd(), "work", spec.name)

    ok = pipeline.run(spec, args.kernel, workdir,
                      from_stage=args.from_stage, to_stage=args.to_stage,
                      verbose=args.verbose, breakdown=args.breakdown,
                      tog=args.tog, tog_grouping=args.tog_grouping)
    return 0 if ok else 1


def main(argv=None):
    ap = parser()
    return run(ap, ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
