#!/usr/bin/env python
"""Turn a TOGSim hardware YAML plus its address map into a target description.

This is the ONE place that knows TOGSim's schema: the compiler reads the JSON
this writes and never sees the YAML, so the machine crosses as data.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorchsim_triton_compiler.contract import target as _target                                # noqa: E402

#: The address map is not in the YAML. Named here so a description still says
#: where the scratchpad is; a caller with its own map writes its own description.
DEFAULT_SPAD_VADDR = 0xD0000000
DEFAULT_SPAD_PADDR = 0x2000000000
DEFAULT_DRAM_BASE = 0x80000000
DEFAULT_DRAM_SIZE = 100 << 30


def from_togsim(yml):
    """A target dict from a decoded TOGSim YAML and the address map above: the matmul unit is the
    YAML's array unit (TOGSim's default SYSTOLIC) lowercased, its queue the key that unit reads.
    The YAML states the scratchpad TWICE, so it is made to agree with itself here."""
    unit = str(yml.get("systolic_array_unit", "SYSTOLIC")).lower()
    if unit not in _target.MATMUL_UNITS:
        raise SystemExit(f"the TOGSim config's systolic_array_unit {unit.upper()} is none of "
                         f"{', '.join(u.upper() for u in _target.MATMUL_UNITS)}")
    queue = _target.MATMUL_UNITS[unit]
    need = ("vpu_num_lanes", "vpu_vector_length_bits", "vpu_spad_size_kb_per_lane", queue.key)
    missing = [k for k in need if k not in yml]
    if missing:
        raise SystemExit(f"the TOGSim config has no {', '.join(missing)}")
    stray = [q.key for u, q in _target.MATMUL_UNITS.items() if u != unit and q.key in yml]
    if stray:
        raise SystemExit(f"the TOGSim config's array unit is {unit.upper()}, whose queue is "
                         f"{queue.key}, and it also states {', '.join(stray)}, another unit's: "
                         f"one of the two was written for some other machine")

    lanes = int(yml["vpu_num_lanes"])
    per_lane_kb = int(yml["vpu_spad_size_kb_per_lane"])

    whole = yml.get("core_spad_size_kb")
    if whole is not None and int(whole) != per_lane_kb * lanes:
        raise SystemExit(
            f"this TOGSim config disagrees with itself about the scratchpad: "
            f"vpu_spad_size_kb_per_lane={per_lane_kb} x vpu_num_lanes={lanes} is "
            f"{per_lane_kb * lanes} kB, but core_spad_size_kb says {whole}. One "
            f"of the two was edited alone.")

    return {
        "vector_lanes": lanes,
        "vlen_bits": int(yml["vpu_vector_length_bits"]),
        "spad_size": per_lane_kb * 1024,
        "spad_vaddr": hex(DEFAULT_SPAD_VADDR),
        "spad_paddr": hex(DEFAULT_SPAD_PADDR),
        "dram_base": hex(DEFAULT_DRAM_BASE),
        "dram_size": DEFAULT_DRAM_SIZE,
        queue.field: int(yml[queue.key]),
        "matmul_unit": unit,
    }


def main():
    """Command line: a TOGSim YAML in, a target description out, refusing a bad one.
    """
    ap = argparse.ArgumentParser(prog="emit_target.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml", help="a TOGSim hardware config")
    ap.add_argument("-o", "--out", help="write here instead of stdout")
    ap.add_argument("--name", help="what to call this target (default: the yaml's stem)")
    args = ap.parse_args()

    import yaml as pyyaml
    with open(args.yaml) as fh:
        got = pyyaml.safe_load(fh)
    if not isinstance(got, dict):
        raise SystemExit(f"{args.yaml}: not a YAML mapping")

    desc = from_togsim(got)
    desc["name"] = args.name or os.path.splitext(os.path.basename(args.yaml))[0]
    desc["provenance"] = {
        "togsim_config": os.path.abspath(args.yaml),
        "address_map": "emit_target.py defaults",
    }

    _target.parse(desc, args.out or "<stdout>")          # refuse to write a bad one
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
