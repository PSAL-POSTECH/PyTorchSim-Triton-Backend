"""In a TOG a kernel's number of programs on an axis is that axis's loop bound, at run time too; on
an axis with no loop it is the value the wrapper passes. Three kernels' TOGs are recorded on two grids.
    python3 tests/passes/grid_extent_cases.py        exit 0 iff every case holds"""
import os
import subprocess
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import pytorchsim_triton_compiler  # noqa: E402,F401
from pytorchsim_triton_compiler.togsim import emit_tog  # noqa: E402
from pytorchsim_triton_compiler.togsim import lower_to_emitc as l2e  # noqa: E402
from pytorchsim_triton_compiler.togsim._mlir_util import walk_ops  # noqa: E402
from pytorchsim_triton_compiler.togsim.build_tog import ir  # noqa: E402
from runtime import tog_record  # noqa: E402

GX, GY, PY = 1, 2, 5
SPAD = 4

_MOD = """
module {{
  func.func @kernel(%x: memref<*xf32>, %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32,
                    %pz: i32) attributes {{togsim.buffer_count = 1 : i64}} {{
    %row = arith.muli %py, {extent} : i32
    %o = arith.index_cast %row : i32 to index
    %sem = arith.constant 0 : index
    "togsim.dma"(%o, %sem) {{dir = 0 : i32, arg_id = 0 : i32, dims = array<i64: 128>,
        strides = array<i64: 1>, elem = "f32", is_async = false, read_bufs = array<i64>,
        write_bufs = array<i64: 0>, indirect = 0 : i32}} : (index, index) -> ()
    return
  }}
}}
"""


def _kernel(extent):
    module = ir.Module.parse(_MOD.format(extent=extent))
    return module, l2e._find_kernel(module)


def _item(extent):
    return l2e.WorkItem(parallel_args=[PY], grid=[extent], key_shifts=[21], grid_args=[GY],
                        fixed_args={GX: 7})


def _one(module, name):
    return [op.operation for op in walk_ops(module.body) if op.operation.name == name][0]


def _dispatched(extent):
    """(how many values the lowered TOG hands a work-item of a dynamic one-axis grid, the axes
    whose extents the WorkItem says follow the id)."""
    module, _ = _kernel(extent)
    item = _item(None)
    l2e.lower_to_emitc(module, SPAD, work_item=item)
    assert module.operation.verify()
    calls = [op.operation for op in walk_ops(module.body) if op.operation.name == "emitc.call_opaque"
             and ir.StringAttr(op.operation.attributes["callee"]).value == l2e.ARGS]
    return len(calls[0].operands), item.extents_handed


def a_static_axis_reads_its_loop_bound_constant():
    module, kernel = _kernel("%gy")
    l2e._materialize_grid_loop(kernel, _item(3), module.context)
    read = _one(module, "arith.muli").operands[1].owner
    bound = _one(module, "scf.for").operands[1]
    assert read.name == "arith.index_cast" and read.operands[0] == bound, read
    assert ir.IntegerAttr(bound.owner.attributes["value"]).value == 3, bound.owner


def a_dynamic_axis_reads_the_shape_arg_its_loop_runs_to():
    module, kernel = _kernel("%gy")
    item = _item(None)
    pending = l2e._materialize_grid_loop(kernel, item, module.context)
    l2e._bind_runtime_bounds(pending, l2e._rewrite_signature(kernel, module.context), item,
                             module.context)
    read = _one(module, "arith.muli").operands[1].owner
    bound = _one(module, "scf.for").operands[1]
    assert read.operands[0] == bound and l2e._const_op(bound) is None, bound.owner
    assert _one(module, "emitc.load").result == bound.owner.operands[0], bound.owner


def a_work_item_is_handed_the_dynamic_extent_it_reads_after_its_id():
    assert _dispatched("%gy") == (2, (0,))


def a_work_item_that_reads_no_extent_is_handed_its_id_alone():
    assert _dispatched("%py") == (1, ())


def an_axis_with_no_loop_reads_the_value_the_wrapper_passes():
    module, kernel = _kernel("%gx")
    l2e._materialize_grid_loop(kernel, _item(None), module.context)
    read = _one(module, "arith.muli").operands[1].owner
    assert read.name == "arith.constant" and ir.IntegerAttr(read.attributes["value"]).value == 7, read


_SCALAR_MOD = """
module {
  func.func @kernel(%x: memref<*xf32>, %a: f32, %m: i32, %n: i32, %gx: i32, %gy: i32, %gz: i32,
                    %px: i32, %py: i32, %pz: i32) attributes {togsim.buffer_count = 1 : i64} {
    %row = arith.muli %py, %n : i32
    %o = arith.index_cast %row : i32 to index
    %sem = arith.constant 0 : index
    "togsim.dma"(%o, %sem) {dir = 0 : i32, arg_id = 0 : i32, dims = array<i64: 128>,
        strides = array<i64: 1>, elem = "f32", is_async = false, read_bufs = array<i64>,
        write_bufs = array<i64: 0>, indirect = 0 : i32} : (index, index) -> ()
    return
  }
}
"""


def a_scalar_is_read_from_shape_args_after_the_grid_and_handed_by_name():
    """%n, the second integer scalar, is shape_args[1 + 1] and is handed after the program id;
    %m is read by nothing and %a, a float, is not carried."""
    module = ir.Module.parse(_SCALAR_MOD)
    item = l2e.WorkItem(parallel_args=[8], grid=[None], key_shifts=[21], grid_args=[5],
                        fixed_args={4: 1, 6: 1}, axes=["y"], scalar_args=[(2, "m", 3), (3, "n", 9)])
    l2e.lower_to_emitc(module, SPAD, work_item=item)
    assert module.operation.verify()
    assert item.scalars_handed == ("n",) and item.extents_handed == (), item.scalars_handed
    reads = [ir.IntegerAttr(op.operation.attributes["value"]).value for op in walk_ops(module.body)
             if op.operation.name == "emitc.constant" and l2e._const_op(op.operation.results[0])
             and any(u.owner.name == "emitc.subscript" for u in op.operation.results[0].uses)]
    assert sorted(reads) == [0, 2], reads
    calls = [op.operation for op in walk_ops(module.body) if op.operation.name == "emitc.call_opaque"
             and ir.StringAttr(op.operation.attributes["callee"]).value == l2e.ARGS]
    assert len(calls[0].operands) == 2, calls[0]


def _work_item(grid, axes):
    """emit_tog.work_item of a kernel with two tensors, one scalar and one constexpr: its grid
    arguments are 3, 4, 5 and its program ids 6, 7, 8."""
    spec = types.SimpleNamespace(name="k", grid=grid, parallel_axes=axes, extra={}, signature={
        "x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"})
    item = emit_tog.work_item(spec)
    return item.parallel_args, item.grid, item.grid_args, item.fixed_args


def every_launch_axis_is_dynamic_and_paired_with_its_own_arguments():
    assert _work_item((4, 3, 2), ("z", "y", "x")) == ([8, 7, 6], [None] * 3, [5, 4, 3], {})
    assert _work_item((4, 3, 2), ("x", "y", "z")) == ([6, 7, 8], [None] * 3, [3, 4, 5], {})


def an_axis_the_launch_does_not_spread_over_is_fixed_at_the_wrapper_s_extent():
    """Its extent is the wrapper's 1, and its program id is the 0 the wrapper hands, both fixed."""
    assert _work_item((5,), ("x",)) == ([6], [None], [3], {4: 1, 5: 1, 7: 0, 8: 0})
    assert _work_item((2, 7), ("y", "x")) == ([7, 6], [None] * 2, [4, 3], {5: 1, 8: 0})


def _recorded(kernel, shape):
    """(the manifest's launch shape, the events recorded on it, the events on `shape`) of the TOG
    of tests/cases/mechanism/<kernel>."""
    from pytorchsim_triton_compiler.contract import kernel_object as ko
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"),
                        os.path.join(ROOT, "tests", "cases", "mechanism", kernel), "--workdir", d,
                        "--tog", "--to", "binary"], check=True, capture_output=True)
        manifest = ko.read(d)
        launch = ko.launch_shape(manifest)
        return (launch, tog_record.record(d, ko.shape_args(manifest)),
                tog_record.record(d, ko.shape_args(manifest, grid=shape)))


def _rows_differ(events, extent, cols):
    """How many dmas of a ptr_batch_index_from_num_programs kernel are off its arithmetic on a grid
    of extents `extent`: load row q = y + z * NY, store row NY * NZ - 1 - q, `cols` columns."""
    wrong = n = 0
    for e in events:
        if e["op"] != "dma":
            continue
        pid = dict(zip(extent, e["iv"]))
        q = pid["y"] + pid["z"] * extent["y"]
        row = q if e["dir"] == 0 else extent["y"] * extent["z"] - 1 - q
        n += 1
        wrong += e["offset"] != row * cols + pid["x"] * 128
    assert n == 2 * extent["x"] * extent["y"] * extent["z"], n
    return wrong


def the_batch_index_follows_the_grid_the_tog_is_run_with():
    launch, first, second = _recorded("ptr/ptr_batch_index_from_num_programs.py", [2, 2, 3])
    assert launch == [2, 3, 4], launch
    assert _rows_differ(first, dict(x=2, y=3, z=4), 256) == 0
    assert _rows_differ(second, dict(x=2, y=2, z=3), 256) == 0


def the_batch_index_follows_the_grid_with_z_outermost():
    launch, first, second = _recorded("ptr/ptr_batch_index_from_num_programs_outermost_z.py",
                                      [3, 2, 4])
    assert launch == [2, 3, 4], launch
    assert _rows_differ(first, dict(z=2, y=3, x=4), 512) == 0
    assert _rows_differ(second, dict(z=3, y=2, x=4), 512) == 0


def the_last_program_s_masked_extent_follows_the_grid_the_tog_is_run_with():
    def stores(events):
        return [e["dims"] for e in events if e["op"] == "dma" and e["dir"] == 1]
    launch, first, second = _recorded("mask/mask_bound_from_num_programs.py", [2])
    assert launch == [3], launch
    assert stores(first) == [[128], [128], [88]], stores(first)
    assert stores(second) == [[128], [88]], stores(second)


CASES = [a_static_axis_reads_its_loop_bound_constant,
         a_dynamic_axis_reads_the_shape_arg_its_loop_runs_to,
         a_work_item_is_handed_the_dynamic_extent_it_reads_after_its_id,
         a_work_item_that_reads_no_extent_is_handed_its_id_alone,
         an_axis_with_no_loop_reads_the_value_the_wrapper_passes,
         a_scalar_is_read_from_shape_args_after_the_grid_and_handed_by_name,
         every_launch_axis_is_dynamic_and_paired_with_its_own_arguments,
         an_axis_the_launch_does_not_spread_over_is_fixed_at_the_wrapper_s_extent,
         the_batch_index_follows_the_grid_the_tog_is_run_with,
         the_batch_index_follows_the_grid_with_z_outermost,
         the_last_program_s_masked_extent_follows_the_grid_the_tog_is_run_with]


def main():
    failed = 0
    for case in CASES:
        try:
            with ir.Context() as ctx, ir.Location.unknown():
                ctx.allow_unregistered_dialects = True
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
