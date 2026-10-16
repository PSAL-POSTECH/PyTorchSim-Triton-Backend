"""The tile-operation-graph (TOG) TOGSim replays, as C++ and a .so, and the compute type of every
tile."""
import json
import os
import sys

from ... import machine
from ...contract import kernel_object
from ...contract.kernel_object import PROGRAM_KEY_SHIFT, TRAILING_GRID_PARAMS
from ...metrics import wallclock

TOG_CPP = "tog.cpp"
TOG_SO = "tog.so"
TILE_TYPES = "tile_types.json"

FILENAME = "timing-tog.json"


def work_item(spec):
    """The WorkItem describing where this kernel's program ids and numbers of programs are; an
    axis the launch does not spread over has the extent the wrapper passes, a constant."""
    from .passes.lower_to_emitc import WorkItem

    sig = spec.signature.values()
    tensors = sum(1 for v in sig if v.startswith("*"))
    scalars = sum(1 for v in sig if not v.startswith("*") and v != "constexpr")
    axes = spec.parallel_axes
    at = {name: tensors + scalars + i for i, name in enumerate(TRAILING_GRID_PARAMS)}
    extent = dict(zip("xyz", kernel_object.grid3(spec.grid)))
    return WorkItem(parallel_args=[at[f"pid{a.upper()}"] for a in axes],
                    grid=[None] * len(axes),
                    key_shifts=[PROGRAM_KEY_SHIFT[a] for a in axes],
                    grid_args=[at[f"grid{a.upper()}"] for a in axes],
                    fixed_args={at[f"grid{a.upper()}"]: n for a, n in extent.items()
                                if a not in axes})


def emit_tog(spec, workdir):
    """Write tog.cpp, tog.so and tile_types.json into the kernel object, and the tile ids into its
    post-vcix IR; `dispatch_extents` names the axes whose extents a work-item is handed after its
    program ids."""
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
        bs.build_skeleton(module, stamped=postvcix)
        types = ct._compute_types(module)
        item = work_item(spec)
        cpp = l2e.emitc_to_cpp(l2e.lower_to_emitc(module, work_item=item))

    cpp_path = os.path.join(workdir, TOG_CPP)
    with open(cpp_path, "w") as fh:
        fh.write(cpp)
    so_path = l2e.compile_so(cpp_path, os.path.join(workdir, TOG_SO))

    types_path = os.path.join(workdir, TILE_TYPES)
    with open(types_path, "w") as fh:
        json.dump({"compute_types": types,
                   "overlap_offset": [ct.overlap_offset(t, machine.VECTORLANE_SIZE) for t in types],
                   "parallel_axes": list(spec.parallel_axes),
                   "dispatch_extents": [spec.parallel_axes[i] for i in item.extents_handed]},
                  fh, indent=2)
        fh.write("\n")

    kernel_object.add_artifact(workdir, "tog_cpp", TOG_CPP)
    kernel_object.add_artifact(workdir, "tog_so", TOG_SO)
    kernel_object.add_artifact(workdir, "tile_types", TILE_TYPES)
    return so_path, types_path


def main(argv):
    """Command line: emit the TOG for one spec, and time the build."""
    import argparse

    from ...contract.kernel_spec import load_spec

    ap = argparse.ArgumentParser(prog="pytorchsim_triton_compiler.lowering.tog.emit_tog",
                                 description=__doc__)
    ap.add_argument("spec", help="kernel spec .py")
    ap.add_argument("workdir")
    args = ap.parse_args(argv[1:])

    spec = load_spec(args.spec)
    wallclock.PROF.label = spec.name
    try:
        with wallclock.PROF.stage("tog"):
            for p in emit_tog(spec, args.workdir):
                print(p)
    finally:
        wallclock.PROF.write(os.path.join(os.path.abspath(args.workdir), FILENAME))
        if wallclock.want_report():
            print(wallclock.PROF.report())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
