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

#: The address map is not in the YAML: PyTorchSim states it in
#: extension_config. Named here so a description generated without
#: PyTorchSim still says where the scratchpad is, rather than guessing.
DEFAULT_SPAD_VADDR = 0xD0000000
DEFAULT_SPAD_PADDR = 0x2000000000
DEFAULT_DRAM_BASE = 0x80000000
DEFAULT_DRAM_SIZE = 100 << 30


def from_togsim(yml, spad_info=None):
    """A target dict from a decoded TOGSim YAML and an optional address map.

    The YAML states the scratchpad TWICE, so it is made to agree with itself here,
    where both halves are in hand, rather than in the compiler that reads one.
    """
    need = ("vpu_num_lanes", "vpu_vector_length_bits", "vpu_spad_size_kb_per_lane",
            "tpu_systolic_queue_entries")
    missing = [k for k in need if k not in yml]
    if missing:
        raise SystemExit(f"the TOGSim config has no {', '.join(missing)}")

    lanes = int(yml["vpu_num_lanes"])
    per_lane_kb = int(yml["vpu_spad_size_kb_per_lane"])

    whole = yml.get("core_spad_size_kb")
    if whole is not None and int(whole) != per_lane_kb * lanes:
        raise SystemExit(
            f"this TOGSim config disagrees with itself about the scratchpad: "
            f"vpu_spad_size_kb_per_lane={per_lane_kb} x vpu_num_lanes={lanes} is "
            f"{per_lane_kb * lanes} kB, but core_spad_size_kb says {whole}. One "
            f"of the two was edited alone.")

    info = spad_info or {}
    return {
        "vector_lanes": lanes,
        "vlen_bits": int(yml["vpu_vector_length_bits"]),
        "spad_size": per_lane_kb * 1024,
        "spad_vaddr": hex(int(info.get("spad_vaddr", DEFAULT_SPAD_VADDR))),
        "spad_paddr": hex(int(info.get("spad_paddr", DEFAULT_SPAD_PADDR))),
        "dram_base": hex(DEFAULT_DRAM_BASE),
        "dram_size": DEFAULT_DRAM_SIZE,
        "systolic_queue_entries": int(yml["tpu_systolic_queue_entries"]),
    }


def _pytorchsim_spad_info(torchsim_dir):
    """PyTorchSim's own address map, or None -- it owns that fact, not us.
    """
    front = os.path.join(torchsim_dir, "PyTorchSimFrontend")
    if not os.path.isdir(front):
        return None
    sys.path.insert(0, front)
    try:
        import extension_config
        return extension_config.CONFIG_SPAD_INFO
    except Exception:
        return None


def main():
    """Command line: a TOGSim YAML in, a target description out, refusing a bad one.
    """
    ap = argparse.ArgumentParser(prog="emit_target.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml", help="a TOGSim hardware config")
    ap.add_argument("-o", "--out", help="write here instead of stdout")
    ap.add_argument("--name", help="what to call this target (default: the yaml's stem)")
    ap.add_argument("--torchsim-dir", default=os.environ.get("TORCHSIM_DIR", ""),
                    help="read the address map from this PyTorchSim checkout")
    args = ap.parse_args()

    import yaml as pyyaml
    with open(args.yaml) as fh:
        got = pyyaml.safe_load(fh)
    if not isinstance(got, dict):
        raise SystemExit(f"{args.yaml}: not a YAML mapping")

    info = _pytorchsim_spad_info(args.torchsim_dir) if args.torchsim_dir else None
    desc = from_togsim(got, info)
    desc["name"] = args.name or os.path.splitext(os.path.basename(args.yaml))[0]
    desc["provenance"] = {
        "togsim_config": os.path.abspath(args.yaml),
        "address_map": "PyTorchSim" if info else "emit_target.py defaults",
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
