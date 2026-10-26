"""pytorchsim-triton-compiler -- compile a Triton kernel for the PyTorchSim NPU: the RISC-V ELF Spike
runs and, with --tog, the tile-operation-graph (TOG) TOGSim simulates. Running them is the caller's --
PyTorchSim, or tests/pytorchsim-compile."""
import argparse
import faulthandler
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
faulthandler.enable()
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from . import toolchain
from . import pipeline
from .contract import failure
from .contract import target as _target
from .contract.kernel_spec import load_spec


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = parser()
    return run(ap, ap.parse_args(argv))


def parser():
    """The driver's arguments. tests/pytorchsim-compile adds its own to the same parser."""
    ap = argparse.ArgumentParser(prog="pytorchsim-triton-compiler", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel", nargs="?",
                    help="kernel file, or 'doctor' or 'target <name|path>' "
                         "(subcommands spelled here, so the stage flags do not apply to them)")
    ap.add_argument("machine", nargs="?",
                    help="after 'target': a machine's name ($TOGSIM_DIR/configs/<name>.yml) or a path")
    names = list(dict.fromkeys(list(pipeline.STAGES) + pipeline.steps()))
    ap.add_argument("--from", dest="from_stage", default="triton-compile", choices=names)
    ap.add_argument("--to", dest="to_stage", default=None, choices=names)
    ap.add_argument("--workdir", default=None,
                    help="where the artifacts go (default: ./work/<kernel name>)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--breakdown", action="store_true",
                    help="print where the wall clock went: stage, pass, tool")
    ap.add_argument("--tog", action="store_true",
                    help="also emit the tile-operation-graph (TOG) TOGSim simulates, C++ and .so")
    ap.add_argument("--list-stages", action="store_true",
                    help="print the stages and steps this build has, then exit")
    return ap


def run(ap, args):
    """Act on parsed arguments: list the stages, run doctor or target, or compile. Every subcommand is
    decided here, so a driver that extends parser() and calls run() has them all."""
    if args.list_stages:
        _list_stages()
        return 0

    if args.kernel is None:
        ap.print_usage(sys.stderr)
        print(f"{ap.prog}: give a kernel file, or 'doctor', or 'target <name|path>'",
              file=sys.stderr)
        return 2

    if args.kernel == "target":
        if args.machine is None:
            ap.error("target needs a machine's name or path")
        return target(args.machine)

    if args.machine is not None:
        ap.error(f"unrecognized arguments: {args.machine}")

    if args.kernel == "doctor":
        return 0 if toolchain.doctor() else 1

    if not os.path.exists(args.kernel):
        print(f"no such kernel file: {args.kernel}", file=sys.stderr)
        return 2

    if args.workdir:
        failure.clear(args.workdir)
    spec = load_spec(args.kernel)
    workdir = args.workdir or os.path.join(os.getcwd(), "work", spec.name)

    ok = pipeline.run(spec, args.kernel, workdir,
                      from_stage=args.from_stage, to_stage=args.to_stage,
                      verbose=args.verbose, breakdown=args.breakdown,
                      tog=args.tog)
    return 0 if ok else 1


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


def target(machine):
    """`target <name|path>`: what the compiler reads of that machine file, as JSON."""
    try:
        t = _target.load(machine)
    except (_target.TargetError, OSError) as e:
        print(f"target: {e}", file=sys.stderr)
        return 1
    print(json.dumps(dict(t.as_dict(), name=t.name, path=t.file), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
