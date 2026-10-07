"""pytorchsim-triton-compiler -- compile a Triton kernel for the PyTorchSim NPU, and optionally run
it on a device and check it against torch."""
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


def main():
    ap = argparse.ArgumentParser(prog="pytorchsim-triton-compiler", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel", nargs="?",
                    help="kernel file, or 'doctor' (a subcommand spelled here, so the "
                         "stage flags do not apply to it)")
    names = list(dict.fromkeys(list(pipeline.STAGES) + pipeline.steps()))
    ap.add_argument("--from", dest="from_stage", default="triton-compile", choices=names)
    ap.add_argument("--to", dest="to_stage", default=None, choices=names)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--breakdown", action="store_true",
                    help="print where the wall clock went: stage, pass, tool")
    ap.add_argument("--trace", action="store_true",
                    help="also emit the TOGSim trace producer, C++ and .so")
    ap.add_argument("--simulate", action="store_true",
                    help="--trace, then gem5 tile cycles and a TOGSim run")
    ap.add_argument("--list-stages", action="store_true",
                    help="print the stages and steps this build has, then exit")
    args = ap.parse_args()

    if args.list_stages:
        for i, s in enumerate(pipeline.STAGES, 1):
            print(f"  {i}. {s:<17} {pipeline.STAGE_DESC[s]}")
            for step in pipeline.STEPS_OF.get(s, []):
                print(f"       {step:<14} {pipeline.STAGE_DESC[step]}")
        return 0

    if args.kernel in (None, "doctor"):
        return 0 if toolchain.doctor() else 1

    if not os.path.exists(args.kernel):
        print(f"no such kernel file: {args.kernel}", file=sys.stderr)
        return 2

    if args.simulate and "simulate" not in pipeline.STAGE_FN:
        print("--simulate runs simulators, which the compiler does not: run it through "
              "tests/torchsim-compile", file=sys.stderr)
        return 2
    spec = load_spec(args.kernel)
    workdir = args.workdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", spec.name)

    ok = pipeline.run(spec, args.kernel, workdir,
                      from_stage=args.from_stage, to_stage=args.to_stage,
                      verbose=args.verbose, breakdown=args.breakdown,
                      trace=args.trace, simulate=args.simulate)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
