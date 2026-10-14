"""A target whose accelerator is vcix_tpuv6 lowers a dot for the matrix unit -- an operand as a vmatprep
and the half that follows it, four pops an input of bf16.
    python3 tests/passes/mxu_matmul_cases.py        exit 0 iff every case holds"""
import collections
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import toolchain  # noqa: E402
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lib_matmul  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6 import lower_matmul as mxu_pass  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

CASES_DIR = os.path.join(ROOT, "tests", "cases", "mechanism")
BF16 = os.path.join(CASES_DIR, "ops", "ops_bf16_matmul.py")          # 128 x 128 tile, K 128 a dot
WIDE = os.path.join(CASES_DIR, "tile", "tile_gemm_wide_tile_grid.py")  # f32, 256 x 256 tile, K 512
EPI = os.path.join(CASES_DIR, "tile", "tile_gemm_epilogue_at_pop.py")  # bf16 512^3, scale/relu/cast
MXU = os.path.join(ROOT, "pytorchsim_triton_compiler", "targets", "tpuv6e.json")


_Loop = collections.namedtuple("_Loop", "names stages")


def _vcix(o):
    return (o.name, ir.IntegerAttr(o.attributes["opcode"]).value,
            ir.IntegerAttr(o.attributes["imm"]).value)


def _stages(o):
    """A vmatpush: a half of the weights staged."""
    return o.name == "vcix.iv" and ir.IntegerAttr(o.attributes["opcode"]).value == 1


def _lowered(kernel, target):
    """09-custom.mlir walked: (the pass dumps' names, every vcix op as (name, opcode, imm), each
    affine.for as a _Loop, every op name in the module, what each op stamped KEPT_APART says, how
    many vpops sit in no loop that also stages weights). The kernel's expect_ir is not read."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                        "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                       env=dict(os.environ, TORCHSIM_COMPILE_TARGET=target,
                                TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        found, loops, free = [], [], [0]
        with ir.Context() as ctx, open(os.path.join(d, "09-custom.mlir")) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())
            everything = walk(module.operation)

            def visit(op, under_weights):
                for region in op.regions:
                    for block in region.blocks:
                        for inner in block.operations:
                            o = inner.operation
                            if o.name.startswith("vcix."):
                                found.append(_vcix(o))
                                free[0] += o.name == "vcix.v.i" and not under_weights
                            if o.name == "affine.for":
                                inside = walk(o)
                                loops.append(_Loop({i.name for i in inside},
                                                   any(_stages(i) for i in inside)))
                                visit(o, under_weights or loops[-1].stages)
                            else:
                                visit(o, under_weights)
            visit(module.operation, False)
            names = {o.name for o in everything}
            kept = [ir.StringAttr(o.attributes[lib_matmul.KEPT_APART]).value
                    for o in everything if lib_matmul.KEPT_APART in o.attributes]
        return sorted(os.listdir(os.path.join(d, "09-passes"))), found, loops, names, kept, free[0]


def a_retired_field_is_refused():
    raw = dict(_target.load(MXU).as_dict(), matmul_unit="mxu")
    try:
        _target.parse(raw, "<case>")
    except _target.TargetError as e:
        assert "matmul_unit" in str(e) and "vcix_<model>" in str(e), e
    else:
        raise AssertionError("matmul_unit was taken")


def a_bf16_dot_of_k_128_stages_the_first_half_and_latches_once():
    dumps, ops, _, _, _, _ = _lowered(BF16, MXU)
    assert "01-tritonpytorchsim-lower-matmul-tpuv6.mlir" in dumps, dumps
    count = collections.Counter(ops)
    # 128 lanes, 16 rows of bf16 a register, K = 128: eight pairs into the first half of the depth
    # ([4:3] = 1, imm 8 | 3), none into the second. M = 128 is eight inputs: the first is emitted
    # on its own and latches ([2]), the seven after it are one loop body that does not.
    assert count[("vcix.iv", 2, 3)] == 8 and count[("vcix.iv", 1, 3 | 8)] == 8, count
    assert not any(name == "vcix.iv" and opcode == 1 and imm < 0 for name, opcode, imm in ops), count
    assert count[("vcix.iv", 2, 3 | 4)] == 2, count
    assert count[("vcix.iv", 0, 3 | 4)] == 1 and count[("vcix.iv", 0, 3)] == 1, count
    assert count[("vcix.v.i", 2, 0)] == 2 and count[("vcix.v.i", 2, 4)] == 2, count
    assert not any(name == "vcix.i" for name, _, _ in ops), "a compute marker on the matrix unit"


def every_half_follows_its_vmatprep():
    _, ops, _, _, _, _ = _lowered(WIDE, MXU)
    pushes = [(opcode, imm) for name, opcode, imm in ops if name == "vcix.iv"]
    assert pushes and len(pushes) % 2 == 0, len(pushes)
    for (prep, prep_imm), (half, half_imm) in zip(pushes[0::2], pushes[1::2]):
        assert prep == 2 and half == (0 if prep_imm & 4 else 1), (prep, prep_imm, half)
        assert prep_imm & 3 == half_imm & 3 == 0, "f32 words are format 0"


def a_panel_stages_its_first_half_then_its_second_and_the_next_vmatmul_latches():
    """f32, K = 512: two K pairs with an upper tile, 16 pushes of 8 rows a half, each pair emitted
    on its own (the first's pops are the first pair's, the second's the stream's last)."""
    _, ops, _, _, _, _ = _lowered(WIDE, MXU)
    panels, staged, after_panel, latched, computed = 0, [], False, 0, 0
    for name, opcode, imm in ops:
        if name != "vcix.iv":
            continue
        if opcode == 1:
            staged.append(imm)
            after_panel = True
        elif opcode == 0:
            if after_panel:
                assert staged == [8] * 16 + [-16] * 16, staged
                panels, staged, after_panel = panels + 1, [], False
                assert imm & 4, "the first vmatmul after a panel does not latch"
                latched += 1
            else:
                assert not (imm & 4), "a vmatmul in the middle of a panel latches"
                computed += 1
    assert panels == latched >= 1 and computed >= 1, (panels, latched, computed)


def a_c_proven_zero_takes_the_first_pair_as_it_is():
    """f32, K = 512, two K pairs, C filled with zeros before the dot: the first pair's pops are
    written as they are, the second's added. (That the fill goes with its loop is held by the
    epilogue case below; this kernel gathers A in loops of its own.)"""
    _, _, loops, _, _, _ = _lowered(WIDE, MXU)
    popping = [loop.names for loop in loops if "vcix.v.i" in loop.names]
    assert any("arith.addf" not in names for names in popping), "every pop is added into C"
    assert any("arith.addf" in names for names in popping), "no pop is added into C"


def later_pairs_are_added_into_c_and_no_input_tests_its_turn():
    _, _, _, names, _, _ = _lowered(WIDE, MXU)
    assert "scf.if" not in names and "arith.remui" not in names, "an input tests its turn at run time"


def the_epilogue_after_a_dot_runs_at_the_last_pair_s_pops():
    """bf16 512^3 scaled, relu'd and cast to bf16 after the dot: the cast is in a loop that pops,
    no loop is left that does not drive the unit, and nothing was kept apart."""
    _, _, loops, _, kept, _ = _lowered(EPI, MXU)
    assert not kept, kept
    assert any({"arith.truncf", "vcix.v.i"} <= loop.names for loop in loops), "no cast at a pop"
    assert loops and all(any(n.startswith("vcix.") for n in loop.names) for loop in loops), \
        "the epilogue stays a loop"


def the_dot_is_one_stream_across_its_n_pairs():
    """bf16 512^3: two N pairs, two K pairs. Were each N pair a stream of its own, every pop would
    sit in the N pairs' loop beside the weights it stages; in one stream the dot's drain comes
    after the last pair, in loops that stage nothing."""
    _, _, _, _, _, free_pops = _lowered(EPI, MXU)
    assert free_pops > 0, "every pop is inside a loop that also stages weights: a stream per N pair"


def the_model_is_given_no_delay_of_its_own():
    """The unit's delay is the model's default, the array's width + depth - 1; its queues are
    the target's, 512 rows, and reach the model."""
    with tempfile.TemporaryDirectory() as d:
        with open(_target.load(MXU).write_model_yaml(os.path.join(d, "m.yml"))) as f:
            keys = yaml.safe_load(f)
        assert "tpu_mxu_delay_cycles" not in keys and keys["vpu_num_lanes"] == 128, keys
        assert keys["tpu_mxu_queue_entries"] == 512 and "tpu_systolic_queue_entries" not in keys, keys


def the_pops_lag_the_pushes_by_the_delay_but_no_more_than_the_queues_hold():
    """At 128 lanes a word a cycle fills the 511-slot line in 64 inputs of 8 words; with queues
    of 1024 rows the lag is the delay's; the v6e target's 512 hold 64 bf16 inputs (f32: the delay's)."""
    assert mxu_pass._lag(128, 8, 16, 1024) == 66
    assert mxu_pass._lag(128, 8, 16, 512) == 64
    assert mxu_pass._lag(128, 8, 8, 512) == 66
    assert mxu_pass._lag(128, 8, 8, 1024) == 66
    assert mxu_pass._lag(128, 8, 16, 256) == 32
    assert mxu_pass._lag(128, 8, 16, 16) == 2
    assert mxu_pass._lag(128, 8, 16, 8) == 1


CASES = [a_retired_field_is_refused,
         a_bf16_dot_of_k_128_stages_the_first_half_and_latches_once,
         every_half_follows_its_vmatprep,
         a_panel_stages_its_first_half_then_its_second_and_the_next_vmatmul_latches,
         a_c_proven_zero_takes_the_first_pair_as_it_is,
         later_pairs_are_added_into_c_and_no_input_tests_its_turn,
         the_epilogue_after_a_dot_runs_at_the_last_pair_s_pops,
         the_dot_is_one_stream_across_its_n_pairs,
         the_model_is_given_no_delay_of_its_own,
         the_pops_lag_the_pushes_by_the_delay_but_no_more_than_the_queues_hold]

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
