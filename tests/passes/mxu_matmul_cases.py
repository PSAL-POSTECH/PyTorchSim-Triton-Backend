"""A target whose matmul_unit is "mxu" lowers a dot for the matrix unit -- an operand as a vmatprep
and the half that follows it, four pops an input of bf16 -- and any other target as it always did.
    python3 tests/passes/mxu_matmul_cases.py        exit 0 iff every case holds"""
import collections
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import toolchain  # noqa: E402
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402

CASES_DIR = os.path.join(ROOT, "tests", "cases", "mechanism")
BF16 = os.path.join(CASES_DIR, "ops", "ops_bf16_matmul.py")          # 128 x 128 tile, K 128 a dot
WIDE = os.path.join(CASES_DIR, "tile", "tile_gemm_wide_tile_grid.py")  # f32, 256 x 256 tile, K 512
SYSTOLIC = os.path.join(ROOT, "tests", "targets", "tpuv6e.json")
MXU = os.path.join(ROOT, "tests", "targets", "tpuv6e_mxu.json")


def _lowered(kernel, target):
    """(the names of the pass dumps, every vcix op of 09-custom.mlir as (name, opcode, imm))."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                        "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                       env=dict(os.environ, TORCHSIM_COMPILE_TARGET=target))
        found = []
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(open(os.path.join(d, "09-custom.mlir")).read())

            def walk(op):
                for region in op.regions:
                    for block in region.blocks:
                        for inner in block.operations:
                            o = inner.operation
                            if o.name.startswith("vcix."):
                                found.append((o.name, ir.IntegerAttr(o.attributes["opcode"]).value,
                                              ir.IntegerAttr(o.attributes["imm"]).value))
                            walk(o)
            walk(module.operation)
        return sorted(os.listdir(os.path.join(d, "09-passes"))), found


def a_target_that_does_not_say_has_the_systolic_array():
    assert _target.load(SYSTOLIC).matmul_unit == "systolic"
    dumps, ops = _lowered(BF16, SYSTOLIC)
    assert "01-torchsim-lower-matmul.mlir" in dumps, dumps
    assert ("vcix.i", 1, 4) in ops, "the systolic array's compute marker is gone"


def a_unit_no_lowering_knows_is_refused():
    raw = dict(_target.load(MXU).as_dict(), matmul_unit="tpu")
    try:
        _target.parse(raw, "<case>")
    except _target.TargetError as e:
        assert "matmul_unit" in str(e), e
    else:
        raise AssertionError("matmul_unit='tpu' was taken")


def a_bf16_dot_is_sixteen_weight_pairs_and_four_pops_an_input():
    dumps, ops = _lowered(BF16, MXU)
    assert "01-torchsim-lower-matmul-mxu.mlir" in dumps, dumps
    count = collections.Counter(ops)
    # 128 lanes: an array of 256 rows, 16 rows of bf16 a register; one N tile, so no upper columns.
    assert count[("vcix.iv", 2, 3)] == 16 and count[("vcix.iv", 1, 3)] == 16, count
    assert count[("vcix.iv", 2, 3 | 4)] == 1 and count[("vcix.iv", 0, 3)] == 1, count
    assert count[("vcix.v.i", 2, 0)] == 2 and count[("vcix.v.i", 2, 4)] == 2, count
    assert not any(name == "vcix.i" for name, _, _ in ops), "a compute marker on the matrix unit"


def every_half_follows_its_vmatprep():
    _, ops = _lowered(WIDE, MXU)
    pushes = [(opcode, imm) for name, opcode, imm in ops if name == "vcix.iv"]
    assert pushes and len(pushes) % 2 == 0, len(pushes)
    for (prep, prep_imm), (half, half_imm) in zip(pushes[0::2], pushes[1::2]):
        assert prep == 2 and half == (0 if prep_imm & 4 else 1), (prep, prep_imm, half)
        assert prep_imm & 3 == half_imm == 0, "f32 words are format 0"


def the_model_is_told_the_delay_the_target_states():
    with tempfile.TemporaryDirectory() as d:
        text = open(_target.load(MXU).write_model_yaml(os.path.join(d, "m.yml"))).read()
        assert "tpu_mxu_delay_cycles: 162\n" in text and "vpu_num_lanes: 128\n" in text, text
        text = open(_target.load(SYSTOLIC).write_model_yaml(os.path.join(d, "s.yml"))).read()
        assert "tpu_mxu" not in text and len(text.splitlines()) == 4, text


CASES = [a_target_that_does_not_say_has_the_systolic_array,
         a_unit_no_lowering_knows_is_refused,
         a_bf16_dot_is_sixteen_weight_pairs_and_four_pops_an_input,
         every_half_follows_its_vmatprep,
         the_model_is_told_the_delay_the_target_states]

if __name__ == "__main__":
    failed = 0
    for case in CASES:
        try:
            case()
            print("PASS ", case.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL ", case.__name__, "--", e)
    sys.exit(1 if failed else 0)
