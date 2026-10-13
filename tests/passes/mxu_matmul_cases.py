"""A target whose matmul_unit is "mxu" lowers a dot for the matrix unit -- an operand as a vmatprep
and the half that follows it, weights staged a half of the depth at a time deepest first and latched
by the next vmatmul -- and any other target as it always did. Exit 0 iff every case holds."""
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


def _lowered(kernel, target, rows=None):
    """(the names of the pass dumps, every vcix op of 09-custom.mlir as (name, opcode, imm)); each
    vmatpush's (half of the depth, B's row offset in the bank) appended to `rows`, in order."""
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
                                if rows is not None and found[-1][:2] == ("vcix.iv", 1):
                                    rows.append(((found[-1][2] & 31) >> 3, _row_offset(o)))
                            walk(o)
            walk(module.operation)
        return sorted(os.listdir(os.path.join(d, "09-passes"))), found


def _row_offset(push):
    """The constant a vmatpush's register is read at past its tile: the last index of the
    transfer_read under it (through an f32 bitcast) is `tile + offset`. None for no read."""
    read = push.operands[0].owner
    if read.name == "arith.bitcast":
        read = read.operands[0].owner
    if read.name != "vector.transfer_read":
        return None
    add = read.operands[len(read.operands) - 2].owner
    return ir.IntegerAttr(add.operands[1].owner.attributes["value"]).value


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


def a_bf16_dot_of_k_128_pushes_the_first_half_only_and_four_pops_an_input():
    """128 lanes, 16 rows of bf16 a register: K = 128 is 8 pushes to the first half, deepest first;
    the first input latches, the loop over the other seven does not; one N tile, no upper half."""
    rows = []
    dumps, ops = _lowered(BF16, MXU, rows)
    assert "01-torchsim-lower-matmul-mxu.mlir" in dumps, dumps
    count = collections.Counter(ops)
    assert count[("vcix.iv", 2, 3)] == 8 and count[("vcix.iv", 1, 3 | 1 << 3)] == 8, count
    assert rows == [(1, words) for words in range(56, -1, -8)], rows
    assert count[("vcix.iv", 2, 3 | 4)] == 2, count
    assert count[("vcix.iv", 0, 3 | 4)] == 1 and count[("vcix.iv", 0, 3)] == 1, count
    assert count[("vcix.v.i", 2, 0)] == 2 and count[("vcix.v.i", 2, 4)] == 2, count
    assert not any(name == "vcix.i" for name, _, _ in ops), "a compute marker on the matrix unit"


def every_half_follows_its_vmatprep():
    _, ops = _lowered(WIDE, MXU)
    pushes = [(opcode, imm) for name, opcode, imm in ops if name == "vcix.iv"]
    assert pushes and len(pushes) % 2 == 0, len(pushes)
    for (prep, prep_imm), (half, half_imm) in zip(pushes[0::2], pushes[1::2]):
        assert prep == 2 and half == (0 if prep_imm & 4 else 1), (prep, prep_imm, half)
        assert prep_imm & 3 == half_imm & 3 == 0, "f32 words are format 0"


def weights_are_staged_a_half_at_a_time_deepest_first_and_the_next_vmatmul_latches():
    """K = 512 of f32, 8 rows a register: each pair of K tiles is 16 pushes to the first half (rows
    120 down to 0), then 16 to the second (248 down to 128); only the vmatmul after them latches."""
    rows = []
    _, ops = _lowered(WIDE, MXU, rows)
    assert all(-16 <= imm < 16 for _, _, imm in ops), "a field of 19:15 is spelled as a SIMM5"
    staged = [(1, r) for r in range(120, -1, -8)] + [(2, r) for r in range(248, 127, -8)]
    assert rows and len(rows) % len(staged) == 0, rows
    assert all(rows[i:i + len(staged)] == staged for i in range(0, len(rows), len(staged))), rows
    last = None
    for name, opcode, imm in ops:
        if name == "vcix.iv" and opcode == 0:
            assert bool(imm & 4) == (last == 1), "a vmatmul latches iff weights were pushed since"
        if name == "vcix.iv" and opcode in (0, 1):
            last = opcode


def the_model_is_given_no_delay_of_its_own():
    """The unit's delay is the model's default, the systolic array's width + depth - 1."""
    with tempfile.TemporaryDirectory() as d:
        text = open(_target.load(MXU).write_model_yaml(os.path.join(d, "m.yml"))).read()
        assert "tpu_mxu" not in text and "vpu_num_lanes: 128\n" in text, text


CASES = [a_target_that_does_not_say_has_the_systolic_array,
         a_unit_no_lowering_knows_is_refused,
         a_bf16_dot_of_k_128_pushes_the_first_half_only_and_four_pops_an_input,
         every_half_follows_its_vmatprep,
         weights_are_staged_a_half_at_a_time_deepest_first_and_the_next_vmatmul_latches,
         the_model_is_given_no_delay_of_its_own]

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
