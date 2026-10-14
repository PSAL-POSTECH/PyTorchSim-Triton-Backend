#!/usr/bin/env python
"""Draft a target from a TOGSim yaml: contract/target.from_togsim on the command line, what the
yaml does not state (isa, the address map) taken from an existing target named by --like."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract import target as _target                                # noqa: E402


def main():
    """A TOGSim yaml in, a draft description out; a bad one is refused, not written."""
    ap = argparse.ArgumentParser(prog="emit_target.py", description=__doc__)
    ap.add_argument("yaml", help="a TOGSim hardware config")
    ap.add_argument("--like", required=True,
                    help="the target (name or path) whose isa and address map the draft takes")
    ap.add_argument("-o", "--out", help="write here instead of stdout")
    ap.add_argument("--name", help="what to call this target (default: the yaml's stem)")
    args = ap.parse_args()

    import yaml as pyyaml
    with open(args.yaml) as fh:
        got = pyyaml.safe_load(fh)
    if not isinstance(got, dict):
        raise SystemExit(f"{args.yaml}: not a YAML mapping")
    like = _target.load(args.like)
    try:
        desc = _target.from_togsim(got, like)
    except _target.TargetError as e:
        raise SystemExit(f"{args.yaml}: {e}")
    desc["name"] = args.name or os.path.splitext(os.path.basename(args.yaml))[0]
    desc["togsim_config"] = os.path.basename(args.yaml)
    desc["provenance"] = {"address_map": f"taken from {like.name}"}

    _target.parse(desc, args.out or "<stdout>")
    text = json.dumps(desc, indent=2) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
