#!/usr/bin/env python
"""Dump what Google's TPU compiler (libtpu) makes of one workload, with no TPU: OUT/hlo, OUT/llo (LLO and
the final VLIW bundles) and OUT/manifest.json. The static schedule only; nothing here reads a dump back.
    setup/toolchain.sh jaxtpu; $TORCHSIM_PREFIX/jax-tpu-venv/bin/python -I tools/dump_tpu_vliw.py OUT matmul 1 384 384"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys

MANIFEST = "manifest.json"
PACKAGES = ("jax", "jaxlib", "libtpu")
DTYPES = {"bf16": "bfloat16", "f32": "float32"}
#: One chip, one host, one slice. A topology of one chip is not divisible by libtpu's default 2x2x1 host.
CHIPS_PER_HOST_BOUNDS = [1, 1, 1]
NUM_SLICES = 1


def matmul(dims):
    """A (M, K) @ B (K, N): the function to compile and the shape of each argument."""
    if len(dims) != 3:
        sys.exit(f"dump_tpu_vliw: matmul takes M K N, got {len(dims)} dims")
    m, k, n = dims
    return (lambda a, b: a @ b), [(m, k), (k, n)]


#: The extension point: a workload name to `dims -> (function, argument shapes)`. Every argument has --dtype.
WORKLOADS = {"matmul": matmul}


def dump_flags(out):
    """The environment that makes XLA and libtpu write their dumps under `out`; set before jax is imported."""
    return {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_dump_hlo_as_text --xla_dump_to={out}/hlo",
        "LIBTPU_INIT_ARGS": (f"--xla_jf_dump_to={out}/llo --xla_jf_dump_hlo_text=true "
                             "--xla_jf_dump_llo_text=true --xla_jf_dump_llo_static_gaps=true "
                             "--xla_jf_emit_annotations=true"),
    }


def compile_for_tpu(fn, shapes, dtype, topology):
    """Lower and compile `fn` for the compile-only `topology`; the devices it was compiled for, as text."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = topologies.get_topology_desc(topology, platform="tpu",
                                           chips_per_host_bounds=CHIPS_PER_HOST_BOUNDS,
                                           num_slices=NUM_SLICES).devices
    sharding = NamedSharding(Mesh(devices, ("x",)), PartitionSpec())
    specs = [jax.ShapeDtypeStruct(s, getattr(jnp, DTYPES[dtype]), sharding=sharding) for s in shapes]
    jax.jit(fn).lower(*specs).compile()
    return [str(d) for d in devices]


def main(argv=None):
    """Refuse a used output directory or a Python without jax + libtpu; else dump, then write the manifest."""
    ap = argparse.ArgumentParser(description="Dump libtpu's HLO, LLO and VLIW bundles for one workload. "
                                             "One run at a time: libtpu holds /tmp/libtpu_lockfile.")
    ap.add_argument("out", help="output directory: must not exist, or be empty")
    ap.add_argument("workload", choices=sorted(WORKLOADS))
    ap.add_argument("dims", nargs="+", type=int, help="the workload's sizes (matmul: M K N)")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--topology", default="v6e=1x1",
                    help="handed to jax.experimental.topologies as it is (default: %(default)s)")
    args = ap.parse_args(argv)

    out = os.path.abspath(args.out)
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        sys.exit(f"dump_tpu_vliw: {out} exists and is not an empty directory")
    fn, shapes = WORKLOADS[args.workload](args.dims)
    missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
    if missing:
        sys.exit(f"dump_tpu_vliw: {sys.executable} has no {', '.join(missing)} -- "
                 f"run setup/toolchain.sh jaxtpu and use its venv's python")

    flags = dump_flags(out)
    os.makedirs(os.path.join(out, "hlo"))
    os.makedirs(os.path.join(out, "llo"))
    os.environ.update(flags)
    devices = compile_for_tpu(fn, shapes, args.dtype, args.topology)

    manifest = {
        "workload": args.workload,
        "dims": args.dims,
        "argument_shapes": shapes,
        "dtype": DTYPES[args.dtype],
        "topology": args.topology,
        "chips_per_host_bounds": CHIPS_PER_HOST_BOUNDS,
        "num_slices": NUM_SLICES,
        "devices": devices,
        "versions": {p: importlib.metadata.version(p) for p in PACKAGES},
        "python": sys.version.split()[0],
        "environment": flags,
    }
    with open(os.path.join(out, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"{out}: {args.workload} {'x'.join(map(str, args.dims))} {args.dtype} on {args.topology}")


if __name__ == "__main__":
    main()
