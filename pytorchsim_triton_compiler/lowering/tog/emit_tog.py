"""The tile-operation-graph (TOG) TOGSim simulates, as C++ and a .so, and the compute type of every
tile."""
import json
import os
import sys

from ... import machine
from . import copy_node
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

    params = kernel_object.parameters(spec)
    entries = kernel_object.scalar_entries(spec)
    handed = kernel_object.launch_scalars(entries)
    axes = spec.parallel_axes
    at = {name: len(params) + i for i, name in enumerate(TRAILING_GRID_PARAMS)}
    extent = dict(zip("xyz", kernel_object.grid3(spec.grid)))
    return WorkItem(parallel_args=[at[f"pid{a.upper()}"] for a in axes],
                    scalar_args=[(params.index(("scalar", entries.index(s))), s["name"], s["default"] or 0)
                                 for s in handed],
                    grid=[None] * len(axes),
                    key_shifts=[PROGRAM_KEY_SHIFT[a] for a in axes],
                    grid_args=[at[f"grid{a.upper()}"] for a in axes],
                    fixed_args={at[f"grid{a.upper()}"]: n for a, n in extent.items()
                                if a not in axes},
                    kernel=spec.name, axes=list(axes), example=[extent[a] for a in axes])


def emit_tog(spec, workdir):
    """Write tog.cpp, tog.so and tile_types.json into the kernel object, and the tile ids into its
    post-vcix IR; `dispatch_extents` names the axes whose extents a work-item is handed after its
    program ids, and `dispatch_scalars` the scalar parameters it is handed after those."""
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
        why = copy_node_why(module)
        held = bs.held_loads(module)
        bs.build_skeleton(module, stamped=postvcix)
        types = ct._compute_types(module)
        ids = _take_held_buffers(module)
        item = work_item(spec)
        cpp = l2e.emitc_to_cpp(l2e.lower_to_emitc(module, work_item=item))
        nodes = [copy_node.emit(workdir, h.arg, h.element_type, h.numel, h.bytes, i) for h, i in zip(held, ids)]

    cpp_path = os.path.join(workdir, TOG_CPP)
    with open(cpp_path, "w") as fh:
        fh.write(cpp)
    so_path = l2e.compile_so(cpp_path, os.path.join(workdir, TOG_SO))

    types_path = os.path.join(workdir, TILE_TYPES)
    with open(types_path, "w") as fh:
        json.dump({"compute_types": types,
                   "overlap_offset": [ct.overlap_offset(t, machine.VECTORLANE_SIZE) for t in types],
                   "parallel_axes": list(spec.parallel_axes),
                   "dispatch_extents": [spec.parallel_axes[i] for i in item.extents_handed],
                   "dispatch_scalars": list(item.scalars_handed)},
                  fh, indent=2)
        fh.write("\n")

    kernel_object.add_artifact(workdir, "tog_cpp", TOG_CPP)
    kernel_object.add_artifact(workdir, "tog_so", TOG_SO)
    kernel_object.add_artifact(workdir, "tile_types", TILE_TYPES)
    kernel_object.set_copy_nodes(workdir, nodes)
    return so_path, types_path, why


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
            so, types, why = emit_tog(spec, args.workdir)
            print(f"copy node: {why}")
            for p in (so, types):
                print(p)
    finally:
        wallclock.PROF.write(os.path.join(os.path.abspath(args.workdir), FILENAME))
        if wallclock.want_report():
            print(wallclock.PROF.report())
    return 0


def copy_node_why(module):
    """What the plan pass said of the module's copy node (p07b_plan_copy_node.COPY_NODE), or why it said
    nothing."""
    from ..passes.p07b_plan_copy_node import COPY_NODE, NAME
    from .passes.build_tog import ir
    attrs = module.operation.attributes
    if COPY_NODE not in attrs:
        return f"no copy node: {NAME} did not run, the module has no torchsim.dot"
    return ir.StringAttr(ir.DictAttr(attrs[COPY_NODE])["why"]).value


def _take_held_buffers(module):
    """The held buffer ids build_skeleton stamped on the kernel, removed from it; none when it holds none."""
    from .passes import togsim_ops as ts
    from .passes.build_tog import _find_kernel, ir
    attrs = _find_kernel(module).operation.attributes
    if ts.ATTR_HELD_BUFFERS not in attrs:
        return []
    got = list(ir.DenseI64ArrayAttr(attrs[ts.ATTR_HELD_BUFFERS]))
    del attrs[ts.ATTR_HELD_BUFFERS]
    return got


if __name__ == "__main__":
    sys.exit(main(sys.argv))
