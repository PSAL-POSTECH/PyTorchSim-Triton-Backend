#!/usr/bin/env python3
"""Refresh the measured v6e rows the vector and attention reports compare against from a tpu-perf-archive checkout,
naming its commit on every row: python3 validation/measured.py ARCHIVE. The rows are also the workloads PyTorchSim
builds its kernel bundle for (validation/pytorchsim)."""
import argparse
import csv
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_REPO = "https://github.com/PSAL-POSTECH/tpu-perf-archive"
LOAD_PROGRAMS = ("max", "sum", "add1", "slice")

#: (the archive's table, the copy here, which of its rows are compared).
PICKS = (
    ("data/v6e/softmax/torchxla.csv", "vector/tpu_v6e_softmax_torchxla.csv",
     lambda r: "grid" in r["source"].split(";")),
    # The same softmax shapes as one fused Pallas kernel at each workload's block: what the vector report compares
    # softmax with (the torch_xla table stays the workload list).
    ("data/v6e/softmax/pallas.csv", "vector/tpu_v6e_softmax_pallas.csv", lambda r: r["source"] == "grid"),
    ("data/v6e/load/torchxla.csv", "vector/tpu_v6e_load_torchxla.csv",
     lambda r: r["source"] == "series" and r["program"] in LOAD_PROGRAMS),
    ("data/v6e/attention/torchxla.csv", "attention/tpu_v6e_attention_torchxla.csv", lambda r: True),
)


def copy_archive(archive):
    """Copy each table's compared rows out of the checkout at `archive`, naming its commit."""
    commit = subprocess.check_output(["git", "-C", archive, "rev-parse", "--short=7", "HEAD"], text=True).strip()
    for src, dst, keep in PICKS:
        with open(os.path.join(archive, src), newline="") as fh:
            rd = csv.DictReader(fh)
            rows = [dict(r, archive_repo=ARCHIVE_REPO, archive_commit=commit) for r in rd if keep(r)]
            cols = rd.fieldnames + ["archive_repo", "archive_commit"]
        with open(os.path.join(HERE, dst), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows of {src} at {commit} -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", help="a tpu-perf-archive checkout")
    copy_archive(ap.parse_args().archive)
