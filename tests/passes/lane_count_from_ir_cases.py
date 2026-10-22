"""The lane count enters once, at the frontend, as the module's "ttg.threads-per-warp" and each
encoding's lane bases; transform checks it against the machine once and every pass reads the IR.
    python3 tests/passes/lane_count_from_ir_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lib_linear_layout as L  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p02_materialize_layouts as p02  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import transform  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import LANES_ATTR as TPW, walk  # noqa: E402
from pytorchsim_triton_compiler.stage import StageError  # noqa: E402

torchsim.register()


def _transform(attrs, body=""):
    """transform.run on a kernel holding `body` whose module says `attrs`: None, or the StageError."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "07-regularized.mlir"), "w") as f:
            f.write("module attributes {%s} {\n  func.func @k() {\n%s    return\n  }\n}\n" % (attrs, body))
        ctx = types.SimpleNamespace(workdir=d, path=lambda n: os.path.join(d, n), lmul=1, loop_lmul={},
                                    spec=types.SimpleNamespace(args=[], grid=(1, 1, 1), extra=()),
                                    log=lambda *_: None)
        try:
            transform.run(ctx)
        except StageError as e:
            return e
    return None


def a_module_made_for_other_lanes_is_refused_at_transform():
    """Breaks if transform takes an IR made for 64 lanes under a 128-lane machine: below transform's
    entry the passes read the lanes from the IR, so nothing after it would notice."""
    assert machine.VECTORLANE_SIZE == 128, machine.VECTORLANE_SIZE
    e = _transform('"%s" = 64 : i32' % TPW)
    assert e is not None and TPW in str(e) and "64" in str(e) and "128" in str(e), e


def a_module_that_does_not_say_its_lanes_is_refused_by_name():
    """Breaks if transform falls back to a default (Triton's 32, or the machine's) for a module with
    no "ttg.threads-per-warp": what the IR was made for is then a guess."""
    e = _transform('"ttg.num-warps" = 1 : i32')
    assert e is not None and TPW in str(e), e


def a_module_made_for_the_machines_lanes_passes_the_check():
    """Breaks if the check refuses the IR every real kernel brings (128 lanes on a 128-lane machine)."""
    e = _transform('"%s" = %d : i32' % (TPW, machine.VECTORLANE_SIZE))
    assert e is None, e


def a_tile_of_other_lanes_than_its_module_says_is_refused_at_transform():
    """Breaks if transform checks only the module's attr: a 64-lane encoding in a module that says
    128 then reaches p02 (which reads encodings) and p00/p04b (which read the attr) disagreeing."""
    l64 = "#ttg.linear<{register = [[64]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>"
    e = _transform('"%s" = %d : i32' % (TPW, machine.VECTORLANE_SIZE),
                   "    %%e = tensor.empty() : tensor<128xf32, %s>\n" % l64)
    assert e is not None and "64 lanes" in str(e) and TPW in str(e), e


def _linear(regs, lanes):
    return "#ttg.linear<{register = %s, lane = %s, warp = [], block = []}>" % (regs, lanes)


L64 = _linear([[64]], [[1], [2], [4], [8], [16], [32]])
REPLICA64 = _linear([[1], [2], [4], [8], [16], [32], [64]], [[0]] * 6)
FILL64 = """module attributes {"ttg.threads-per-warp" = 64 : i32} {
  func.func @f(%d: memref<128xf32>, %x: f32) {
    %c0 = arith.constant 0 : index
    %e = tensor.empty() : tensor<128xf32, L64>
    %f = linalg.fill ins(%x : f32) outs(%e : tensor<128xf32, L64>) -> tensor<128xf32, L64>
    "torchsim.transfer"(%d, %c0, %f, %c0, %c0) <{dma_kind = "MVOUT", dram_arg = 0 : i64, dram_stride = [1],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}>
        : (memref<128xf32>, index, tensor<128xf32, L64>, index, index) -> ()
    return
  }
}""".replace("L64", L64)
REDUCE64 = """module attributes {"ttg.threads-per-warp" = 64 : i32} {
  func.func @f(%x: f32) -> f32 {
    %e = tensor.empty() : tensor<128xf32, R64>
    %t = linalg.fill ins(%x : f32) outs(%e : tensor<128xf32, R64>) -> tensor<128xf32, R64>
    %a = bufferization.alloc_tensor() : tensor<f32>
    %i = tensor.insert %x into %a[] : tensor<f32>
    %r = linalg.reduce ins(%t : tensor<128xf32, R64>) outs(%i : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %m = arith.mulf %in, %init : f32
        linalg.yield %m : f32
      }
    %v = tensor.extract %r[] : tensor<f32>
    return %v : f32
  }
}""".replace("R64", REPLICA64)
CONVERT = """module attributes {"ttg.threads-per-warp" = 64 : i32} {
  func.func @f(%x: f32) {
    %e = tensor.empty() : tensor<128xf32, L64>
    %c = "ttg.convert_layout"(%e) : (tensor<128xf32, L64>) -> tensor<128xf32, L128>
    return
  }
}""".replace("L64", L64).replace("L128", _linear([], [[1], [2], [4], [8], [16], [32], [64]]))


def _parsed(text):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        return ctx, ir.Module.parse(text)


def a_dim_on_the_irs_lanes_wrapped_into_registers_lays_out_as_it_is():
    """Breaks if physical_perm asks whether a dim fills the lanes of the MACHINE (128) instead of
    its encoding's (64): p02 then cuts a [128] tile on 64 lanes, two laps a lane, into [64, 2]."""
    assert machine.VECTORLANE_SIZE == 128, machine.VECTORLANE_SIZE
    ctx, m = _parsed(FILL64)
    with ctx, ir.Location.unknown():
        perm = L.physical_perm(L.get_bases(ir.Attribute.parse(L64)), [128])
        assert perm == [0], perm
        p02.run(m)
        assert m.operation.verify()
        shapes = [list(ir.RankedTensorType(r.type).shape) for o in walk(m.operation)
                  for r in o.results if isinstance(r.type, ir.RankedTensorType)]
        assert shapes and all(s == [128] for s in shapes), shapes


def a_rank0_replica_takes_the_lanes_of_the_tile_it_folds():
    """Breaks if the rank-0 init of a reduce that folds a 64-lane replica is given the machine's
    lanes (7 lane bases) instead of its input's (6): the reduce's input and init then disagree."""
    ctx, m = _parsed(REDUCE64)
    with ctx, ir.Location.unknown():
        red = next(o for o in walk(m.operation) if o.name == "linalg.reduce")
        with ir.InsertionPoint(red):
            p02._cut_reduce(red)
        inits = [o for o in walk(m.operation) if o.name == "tensor.expand_shape"
                 and not ir.RankedTensorType(o.operands[0].type).shape]
        assert inits, "no rank-0 init was cut"
        got = [len(L.get_bases(ir.RankedTensorType(o.results[0].type).encoding)[1]) for o in inits]
        assert got == [6], got


def tiles_of_two_lane_counts_meeting_are_refused_by_name():
    """Breaks if p02 cuts a convert_layout between a 64-lane and a 128-lane tile as if the lanes
    agreed: the common form then takes one side's lanes for both."""
    ctx, m = _parsed(CONVERT)
    with ctx, ir.Location.unknown():
        cv = next(o for o in walk(m.operation) if o.name == "ttg.convert_layout")
        try:
            with ir.InsertionPoint(cv):
                p02._cut_convert(cv)
        except p02.DigitRefused as e:
            assert "lane bases" in str(e), e
            return
        raise AssertionError("cut without refusal")


def _cut_refused(text, name, cut):
    """`cut` of the first `name` op in `text`: the DigitRefused message, or None."""
    ctx, m = _parsed(text)
    with ctx, ir.Location.unknown():
        op = next(o for o in walk(m.operation) if o.name == name)
        try:
            with ir.InsertionPoint(op):
                cut(op)
        except p02.DigitRefused as e:
            return str(e)
    return None


R0_7 = _linear([], [[]] * 7)
REPLICA128_2D = _linear([[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]],
                        [[0, 0]] * 7)
BROADCAST = """module attributes {"ttg.threads-per-warp" = 64 : i32} {
  func.func @f(%x: tensor<128xf32, R64>) {
    %e = tensor.empty() : tensor<4x128xf32, R128>
    %b = linalg.broadcast ins(%x : tensor<128xf32, R64>) outs(%e : tensor<4x128xf32, R128>) dimensions = [0]
    return
  }
}""".replace("R64", REPLICA64).replace("R128", REPLICA128_2D)


def a_lead_replica_meeting_an_init_of_other_lanes_is_refused_by_name():
    """Breaks if p02 compares lane bases only when a reduce PROMOTES its input to lead: an input
    already lead (a 64-lane replica) with a 128-lane rank-0 init is then cut, the init's lanes lost."""
    got = _cut_refused(REDUCE64.replace("tensor<f32>", "tensor<f32, R0>").replace("R0", R0_7),
                       "linalg.reduce", p02._cut_reduce)
    assert got is not None and "lane bases" in got, got


def a_lead_broadcast_of_a_replica_of_other_lanes_is_refused_by_name():
    """Breaks if p02 compares lane bases only when a broadcast PROMOTES its result to lead: a
    128-lane replica result of a 64-lane replica source is then cut as if the lanes agreed."""
    got = _cut_refused(BROADCAST, "linalg.broadcast", p02._cut_broadcast)
    assert got is not None and "lane bases" in got, got


CASES = [a_module_made_for_other_lanes_is_refused_at_transform,
         a_module_that_does_not_say_its_lanes_is_refused_by_name,
         a_module_made_for_the_machines_lanes_passes_the_check,
         a_tile_of_other_lanes_than_its_module_says_is_refused_at_transform,
         a_dim_on_the_irs_lanes_wrapped_into_registers_lays_out_as_it_is,
         a_rank0_replica_takes_the_lanes_of_the_tile_it_folds,
         tiles_of_two_lane_counts_meeting_are_refused_by_name,
         a_lead_replica_meeting_an_init_of_other_lanes_is_refused_by_name,
         a_lead_broadcast_of_a_replica_of_other_lanes_is_refused_by_name]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
