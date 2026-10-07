"""The TOGSim trace producer's C++, and the compute type of every tile."""
import json
import os
import sys

from ... import machine
from ...contract import kernel_object
from ...contract.kernel_object import TRAILING_GRID_PARAMS
from ...metrics import wallclock

TRACE_CPP = "trace.cpp"
TRACE_SO = "trace.so"
TILE_TYPES = "tile_types.json"

FILENAME = "timing-trace.json"


def work_item(spec):
    """The WorkItem describing where this kernel's program ids are."""
    from .passes.lower_to_emitc import WorkItem

    sig = spec.signature.values()
    tensors = sum(1 for v in sig if v.startswith("*"))
    scalars = sum(1 for v in sig if not v.startswith("*") and v != "constexpr")
    axes = spec.parallel_axes
    base = tensors + scalars
    return WorkItem(parallel_args=[base + TRAILING_GRID_PARAMS.index(f"pid{a.upper()}")
                                   for a in axes],
                    grid=[None] * len(axes))


def build_trace(spec, workdir):
    """Write trace.cpp, trace.so and tile_types.json into the kernel object."""
    import pytorchsim_triton_compiler  # noqa: F401

    from .passes import build_skeleton as bs
    from .passes import cycle_table as ct
    from .passes import lower_to_emitc as l2e
    from .passes.build_tog import ir

    postvcix = kernel_object.artifact(workdir, "post_vcix_ir")

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(postvcix).read(), ctx)
        bs.build_skeleton(module)
        types = ct._compute_types(module)
        cpp = l2e.emitc_to_cpp(l2e.lower_to_emitc(module, work_item=work_item(spec)))

    cpp_path = os.path.join(workdir, TRACE_CPP)
    with open(cpp_path, "w") as fh:
        fh.write(cpp)
    so_path = l2e.compile_so(cpp_path, os.path.join(workdir, TRACE_SO))

    types_path = os.path.join(workdir, TILE_TYPES)
    with open(types_path, "w") as fh:
        json.dump({"compute_types": types,
                   "overlap_offset": [ct.overlap_offset(t, machine.VECTORLANE_SIZE) for t in types],
                   "parallel_axes": list(spec.parallel_axes)}, fh, indent=2)
        fh.write("\n")

    kernel_object.add_artifact(workdir, "trace_cpp", TRACE_CPP)
    kernel_object.add_artifact(workdir, "trace_so", TRACE_SO)
    kernel_object.add_artifact(workdir, "tile_types", TILE_TYPES)
    return so_path, types_path


def main(argv):
    """Command line: build the trace producer for one spec, and time the build."""
    import argparse

    from ...contract.kernel_spec import load_spec

    ap = argparse.ArgumentParser(prog="pytorchsim_triton_compiler.lowering.tog.trace",
                                 description=__doc__)
    ap.add_argument("spec", help="kernel spec .py")
    ap.add_argument("workdir")
    args = ap.parse_args(argv[1:])

    spec = load_spec(args.spec)
    wallclock.PROF.label = spec.name
    try:
        with wallclock.PROF.stage("trace"):
            for p in build_trace(spec, args.workdir):
                print(p)
    finally:
        wallclock.PROF.write(os.path.join(os.path.abspath(args.workdir), FILENAME))
        if wallclock.want_report():
            print(wallclock.PROF.report())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
