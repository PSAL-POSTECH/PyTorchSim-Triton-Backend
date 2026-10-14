"""A systolic stream's push and pop reads the bank without a bounds check where the subtile grid
shows they stay inside it; 22 cycles a push went to that check (the 2026-10 TPU v6e comparison).
    python3 tests/passes/stream_bounds_cases.py        exit 0 iff every case holds"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import toolchain  # noqa: E402
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402

# Cases of torchsim-lower-matmul, so a machine of the systolic array: the repo ships no systolic
# target, so tpuv6e's numbers restated for it, for this process and the compiles it runs.
_HOLD = tempfile.TemporaryDirectory()
os.environ["TORCHSIM_COMPILE_TARGET"] = os.path.join(_HOLD.name, "systolic.json")
with open(os.environ["TORCHSIM_COMPILE_TARGET"], "w") as _f:
    json.dump(_target.load(os.path.join(ROOT, "tests", "targets", "tpuv6e.json"))
              .restated_for("systolic"), _f)

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p08_lower_matmul as p08  # noqa: E402

KERNEL = os.path.join(ROOT, "tests", "cases", "mechanism", "tile", "tile_gemm_streamed_subtiles.py")


def _operand(depth, width):
    v = ir.Operation.create("test.bank", results=[ir.MemRefType.get([1, depth], ir.F32Type.get())]).result
    return p08._GemmOperand(v, ir.VectorType.get([width], ir.F32Type.get()), None, None)


def the_last_vector_that_ends_on_the_bank_end_is_inside():
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        assert p08._in_bank(_operand(1024, 32), 2, 512, 2, 256, 224)


def one_element_past_the_bank_end_is_outside():
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        assert not p08._in_bank(_operand(1023, 32), 2, 512, 2, 256, 224)


def _push_reads(module):
    """The vector.transfer_read each vcix push reads its row from."""
    out = []

    def source(v):
        op = v.owner
        while op.name in ("arith.bitcast", "arith.select"):
            op = op.operands[1 if op.name == "arith.select" else 0].owner
        return op

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    if inner.operation.name == "vcix.iv":
                        out.append(source(inner.operation.operands[0]))
                    walk(inner.operation)
    walk(module.operation)
    return out


def every_push_of_a_whole_grid_stream_reads_without_a_bounds_check():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), KERNEL,
                        "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                       env=dict(os.environ, TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            m = ir.Module.parse(open(os.path.join(d, "09-passes", "01-torchsim-lower-matmul.mlir")).read())
            reads = _push_reads(m)
            assert reads, "no vcix push found"
            for r in reads:
                assert r.name == "vector.transfer_read", r.name
                flags = [ir.BoolAttr(x).value for x in ir.ArrayAttr(r.attributes["in_bounds"])]
                assert all(flags), f"a push read keeps its bounds check: {r}"


CASES = [the_last_vector_that_ends_on_the_bank_end_is_inside,
         one_element_past_the_bank_end_is_outside,
         every_push_of_a_whole_grid_stream_reads_without_a_bounds_check]


def main():
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
