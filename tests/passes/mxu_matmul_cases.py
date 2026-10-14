"""A target whose matmul_unit is "mxu" lowers a dot for the matrix unit -- an operand as a vmatprep
and the half that follows it, four pops an input of bf16 -- and any other target as it always did.
    python3 tests/passes/mxu_matmul_cases.py        exit 0 iff every case holds"""
import collections
import os
import re
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
from pytorchsim_triton_compiler.lowering.passes import p08_lower_matmul as p08  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p08_lower_matmul_mxu as mxu_pass  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

CASES_DIR = os.path.join(ROOT, "tests", "cases", "mechanism")
BF16 = os.path.join(CASES_DIR, "ops", "ops_bf16_matmul.py")          # 128 x 128 tile, K 128 a dot
WIDE = os.path.join(CASES_DIR, "tile", "tile_gemm_wide_tile_grid.py")  # f32, 256 x 256 tile, K 512
EPI = os.path.join(CASES_DIR, "tile", "tile_gemm_epilogue_at_pop.py")  # bf16 512^3, scale/relu/cast
# bf16 2048 x 384 x 1152 as two dots of 1024 rows an N window of 128, both with the window's B
SHARED_B = os.path.join(ROOT, "validation", "gemm", "kernels", "tk_2048x384x1152_a_w128c1024.py")
# bf16, K = 384 walked in blocks of 128 (acc += dot), N = 384 in a block of 512, cast after the loop
K_LOOP = os.path.join(CASES_DIR, "tile", "tile_gemm_k_loop_masked_n.py")
SYSTOLIC = os.path.join(ROOT, "tests", "targets", "tpuv6e.json")
MXU = os.path.join(ROOT, "tests", "targets", "tpuv6e_mxu.json")       # two arrays, as the v6e has
_TARGETS = tempfile.TemporaryDirectory()


def _target_of(arrays):
    """The MXU target with `arrays` arrays, as a file."""
    import json
    path = os.path.join(_TARGETS.name, f"mxu{arrays}.json")
    with open(path, "w") as f:
        json.dump(dict(_target.load(MXU).as_dict(), matmul_units=arrays), f)
    return path


MXU1 = _target_of(1)


_Loop = collections.namedtuple("_Loop", "names stages")


def _vcix(o):
    return (o.name, ir.IntegerAttr(o.attributes["opcode"]).value,
            ir.IntegerAttr(o.attributes["imm"]).value)


def _array(o):
    """(name, opcode, the array the op names): sf.vc.iv's rd, sf.vc.v.i's rs2."""
    field = "rs2" if o.name == "vcix.v.i" else "rd"
    return (o.name, ir.IntegerAttr(o.attributes["opcode"]).value,
            ir.IntegerAttr(o.attributes[field]).value if field in o.attributes else 0)


def _stages(o):
    """A vmatpush: a half of the weights staged."""
    return o.name == "vcix.iv" and ir.IntegerAttr(o.attributes["opcode"]).value == 1


def _lowered(kernel, target):
    """09-custom.mlir walked: (the pass dumps' names, every vcix op as (name, opcode, imm), each
    affine.for as a _Loop, every op name in the module, what each op stamped KEPT_APART says, how
    many vpops sit in no loop that also stages weights, every vcix op as (name, opcode, array),
    how many vector stores there are). The kernel's expect_ir is not read."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                        "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                       env=dict(os.environ, TORCHSIM_COMPILE_TARGET=target,
                                TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        found, loops, free, arrays, stores = [], [], [0], [], [0]
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
                                arrays.append(_array(o))
                                free[0] += o.name == "vcix.v.i" and not under_weights
                            stores[0] += o.name == "vector.transfer_write"
                            if o.name == "affine.for":
                                inside = walk(o)
                                loops.append(_Loop({i.name for i in inside},
                                                   any(_stages(i) for i in inside)))
                                visit(o, under_weights or loops[-1].stages)
                            else:
                                visit(o, under_weights)
            visit(module.operation, False)
            names = {o.name for o in everything}
            kept = [ir.StringAttr(o.attributes[p08.KEPT_APART]).value
                    for o in everything if p08.KEPT_APART in o.attributes]
        return (sorted(os.listdir(os.path.join(d, "09-passes"))), found, loops, names, kept, free[0],
                arrays, stores[0])


def a_target_that_does_not_say_has_the_systolic_array():
    assert _target.load(SYSTOLIC).matmul_unit == "systolic"
    dumps, ops, _, _, _, _, _, _ = _lowered(BF16, SYSTOLIC)
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


def a_bf16_dot_of_k_128_stages_the_first_half_and_latches_once():
    dumps, ops, _, _, _, _, _, _ = _lowered(BF16, MXU)
    assert "01-torchsim-lower-matmul-mxu.mlir" in dumps, dumps
    count = collections.Counter(ops)
    # 128 lanes, 16 rows of bf16 a register, K = 128 a block: eight pairs into the first half of
    # the depth ([4:3] = 1, imm 8 | 3), none into the second. The kernel walks K in two such blocks,
    # a loop whose first and last trip come out of it (_peel_k_loops): each is a dot of its own.
    # M = 128 is eight inputs a dot: the first is emitted on its own and latches ([2]), the seven
    # after it are one loop body that does not.
    dots = 2
    assert count[("vcix.iv", 2, 3)] == 8 * dots and count[("vcix.iv", 1, 3 | 8)] == 8 * dots, count
    assert not any(name == "vcix.iv" and opcode == 1 and imm < 0 for name, opcode, imm in ops), count
    assert count[("vcix.iv", 2, 3 | 4)] == 2 * dots, count
    assert count[("vcix.iv", 0, 3 | 4)] == dots and count[("vcix.iv", 0, 3)] == dots, count
    assert count[("vcix.v.i", 2, 0)] == 2 * dots and count[("vcix.v.i", 2, 4)] == 2 * dots, count
    assert not any(name == "vcix.i" for name, _, _ in ops), "a compute marker on the matrix unit"


def every_half_follows_its_vmatprep():
    _, ops, _, _, _, _, _, _ = _lowered(WIDE, MXU)
    pushes = [(opcode, imm) for name, opcode, imm in ops if name == "vcix.iv"]
    assert pushes and len(pushes) % 2 == 0, len(pushes)
    for (prep, prep_imm), (half, half_imm) in zip(pushes[0::2], pushes[1::2]):
        assert prep == 2 and half == (0 if prep_imm & 4 else 1), (prep, prep_imm, half)
        assert prep_imm & 3 == half_imm & 3 == 0, "f32 words are format 0"


def a_panel_stages_its_first_half_then_its_second_and_the_next_vmatmul_latches():
    """f32, K = 512, one array: two K pairs with an upper tile, 16 pushes of 8 rows a half, each
    pair emitted on its own (the first's pops are the first pair's, the second's the stream's
    last)."""
    _, ops, _, _, _, _, _, _ = _lowered(WIDE, MXU1)
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
    _, _, loops, _, _, _, _, _ = _lowered(WIDE, MXU)
    popping = [loop.names for loop in loops if "vcix.v.i" in loop.names]
    assert any("arith.addf" not in names for names in popping), "every pop is added into C"
    assert any("arith.addf" in names for names in popping), "no pop is added into C"


def later_pairs_are_added_into_c_and_no_input_tests_its_turn():
    _, _, _, names, _, _, _, _ = _lowered(WIDE, MXU)
    assert "scf.if" not in names and "arith.remui" not in names, "an input tests its turn at run time"


def the_epilogue_after_a_dot_runs_at_the_last_pair_s_pops():
    """bf16 512^3 scaled, relu'd and cast to bf16 after the dot: the cast is in a loop that pops,
    no loop is left that does not drive the unit, and nothing was kept apart."""
    _, _, loops, _, kept, _, _, _ = _lowered(EPI, MXU)
    assert not kept, kept
    assert any({"arith.truncf", "vcix.v.i"} <= loop.names for loop in loops), "no cast at a pop"
    assert loops and all(any(n.startswith("vcix.") for n in loop.names) for loop in loops), \
        "the epilogue stays a loop"


def the_dot_is_one_stream_across_its_n_pairs():
    """bf16 512^3: two N pairs, two K pairs. Were each N pair a stream of its own, every pop would
    sit in the N pairs' loop beside the weights it stages; in one stream the dot's drain comes
    after the last pair, in loops that stage nothing."""
    _, _, _, _, _, free_pops, _, _ = _lowered(EPI, MXU)
    assert free_pops > 0, "every pop is inside a loop that also stages weights: a stream per N pair"


def the_model_is_given_no_delay_of_its_own():
    """The unit's delay is the model's default, the systolic array's width + depth - 1; its
    queues are the target's, which hold what the delay lets out at two rows a cycle."""
    with tempfile.TemporaryDirectory() as d:
        with open(_target.load(MXU).write_model_yaml(os.path.join(d, "m.yml"))) as f:
            keys = yaml.safe_load(f)
        assert "tpu_mxu_delay_cycles" not in keys and keys["vpu_num_lanes"] == 128, keys
        assert keys["tpu_systolic_queue_entries"] == 1024 and keys["tpu_mxu_units"] == 2, keys
        with open(_target.load(SYSTOLIC).write_model_yaml(os.path.join(d, "s.yml"))) as f:
            keys = yaml.safe_load(f)
        assert keys["tpu_systolic_queue_entries"] == 256 and "tpu_mxu_units" not in keys, keys


def two_arrays_take_the_k_pairs_by_turns_and_add_their_pops_in_registers():
    """f32, K = 512 on two arrays: the two K pairs are one group, a panel on each array, every
    input goes to both, and each register of the result is the two arrays' pops added before C
    is touched -- two pops a store, where one array has one."""
    _, _, _, _, _, _, arrays, stores = _lowered(WIDE, MXU)
    by = collections.Counter(arrays)
    assert by[("vcix.iv", 1, 0)] == by[("vcix.iv", 1, 1)] > 0, by     # vmatpush
    assert by[("vcix.iv", 0, 0)] == by[("vcix.iv", 0, 1)] > 0, by     # vmatmul
    assert by[("vcix.v.i", 2, 0)] == by[("vcix.v.i", 2, 1)] > 0, by   # vpop
    pops = [array for name, opcode, array in arrays if (name, opcode) == ("vcix.v.i", 2)]
    assert pops == [0, 1] * (len(pops) // 2), pops[:8]
    assert len(pops) == 2 * stores, (len(pops), stores)
    _, _, _, _, _, _, arrays, stores = _lowered(WIDE, MXU1)
    pops = [array for name, opcode, array in arrays if (name, opcode) == ("vcix.v.i", 2)]
    assert set(pops) == {0} and len(pops) == stores, (len(pops), stores)


def a_dot_of_one_k_tile_uses_the_first_array_alone():
    """bf16, K = 128 on two arrays: one K pair, so one group of one panel, on array 0; nothing
    names array 1. (A Triton dot's K is a power of two, so an odd last K tile only ever comes
    alone: the lowering's mixed group of a pair and an odd tile has no kernel to come from.)"""
    _, _, _, _, _, _, arrays, _ = _lowered(BF16, MXU)
    named = {array for name, opcode, array in arrays if name == "vcix.iv" or opcode == 2}
    assert named == {0}, named


def the_sample_gem5_times_drives_the_first_array_alone():
    """The arrays work in the same cycles, so the timed sample keeps array 0's instructions: an
    array 1 vmatmul goes with the read only it took, its vpop becomes zeros, and the add of the two
    arrays' rows stays for the vector unit to do."""
    from pytorchsim_triton_compiler.lowering.tog.passes.one_array import keep_first_array
    text = """
func.func @f(%a: memref<1x64xi32>, %c: memref<1x64xf32>, %vl: i64) {
  %i = arith.constant 0 : index
  %p = arith.constant 0 : i32
  %z = arith.constant 0.0 : f32
  %r0 = vector.transfer_read %a[%i, %i], %p {in_bounds = [true]} : memref<1x64xi32>, vector<8xi32>
  "vcix.iv"(%r0, %vl) {opcode = 0 : i64, imm = 3 : i64, rd = 0 : i64} : (vector<8xi32>, i64) -> ()
  %r1 = vector.transfer_read %a[%i, %i], %p {in_bounds = [true]} : memref<1x64xi32>, vector<8xi32>
  "vcix.iv"(%r1, %vl) {opcode = 0 : i64, imm = 3 : i64, rd = 1 : i64} : (vector<8xi32>, i64) -> ()
  %p0 = "vcix.v.i"(%vl) {opcode = 2 : i64, imm = 0 : i64, rs2 = 0 : i64} : (i64) -> vector<8xi32>
  %p1 = "vcix.v.i"(%vl) {opcode = 2 : i64, imm = 0 : i64, rs2 = 1 : i64} : (i64) -> vector<8xi32>
  %f0 = arith.bitcast %p0 : vector<8xi32> to vector<8xf32>
  %f1 = arith.bitcast %p1 : vector<8xi32> to vector<8xf32>
  %s = arith.addf %f0, %f1 : vector<8xf32>
  vector.transfer_write %s, %c[%i, %i] {in_bounds = [true]} : vector<8xf32>, memref<1x64xf32>
  return
}"""
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(text)
        assert keep_first_array(module) == 2
        left = collections.Counter(o.name for o in walk(module.operation))
        fields = [_array(o) for o in walk(module.operation) if o.name.startswith("vcix.")]
    assert fields == [("vcix.iv", 0, 0), ("vcix.v.i", 2, 0)], fields
    assert left["vector.transfer_read"] == 1 and left["arith.addf"] == 1, left


def a_dot_that_follows_one_with_the_same_weights_stages_none_again():
    """Two dots in a block with one B, nothing writing it between: the arrays hold the first's
    panel when the second starts, so only the first stages it, and both compute their 64 inputs on
    each array."""
    _, _, _, _, _, _, arrays, _ = _lowered(SHARED_B, MXU)
    by = collections.Counter(arrays)
    # K = 384 in a block of 512: three live K tiles, a pair on array 0 (both halves of the depth,
    # 16 pushes) and the third on array 1 (the first half alone, 8); once for both dots.
    assert (by[("vcix.iv", 1, 0)], by[("vcix.iv", 1, 1)]) == (16, 8), by
    assert by[("vcix.iv", 0, 0)] == by[("vcix.iv", 0, 1)] > 0, by


def _facts_of(kernel, target):
    """The pass's step 2 on `kernel`'s 08-transformed.mlir, after step 1: each dot's _Facts, and
    the module's text."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                        "--workdir", d, "--to", "transform"], check=True, capture_output=True,
                       env=dict(os.environ, TORCHSIM_COMPILE_TARGET=target,
                                TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        t = _target.load(target)
        with ir.Context() as ctx, ir.Location.unknown(), \
                open(os.path.join(d, "08-transformed.mlir")) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())
            from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import index_bank_casts
            index_bank_casts(module)
            mxu_pass._peel_k_loops(module)
            dots = [o for o in walk(module.operation) if o.name in mxu_pass.MATMUL_OPS]
            facts = [mxu_pass._facts(o, dots[i - 1] if i else None, t.vector_lanes, t.vlen_bits,
                                     (1, 1, 1)) for i, o in enumerate(dots)]
            return facts, str(module)


def a_k_loop_gives_its_first_and_last_trips_apart():
    """K = 384 in blocks of 128 is three trips: the first and the last come out of the loop, the
    middle one stays. Only the first takes a C its fill proves zero; each trip's K block is one
    lane tile, and N = 384 is three of the block's four lane tiles in every trip."""
    facts, text = _facts_of(K_LOOP, MXU)
    assert len(facts) == 3 and text.count("scf.for") == 1, (len(facts), text.count("scf.for"))
    # The middle trips' loop starts at 0: the TOG's sample runs a loop's trip from its start, and
    # one that started past it ran none, so its markers never fired.
    assert re.search(r"scf\.for %\w+ = %c0_i32\w* to", text), [l for l in text.splitlines() if "scf.for" in l]
    assert [f.init is not None for f in facts] == [True, False, False], facts
    assert [f.live for f in facts] == [(3, 1)] * 3, facts


def a_k_loop_s_fill_goes_and_its_cast_runs_at_the_last_pops():
    """Lowered: no zero is stored to C any more, and no loop casts f32 to bf16 apart from the pops:
    every loop that truncates also pops."""
    _, _, loops, names, _, _, _, _ = _lowered(K_LOOP, MXU)
    assert "arith.truncf" in names, names
    apart = [lp for lp in loops if "arith.truncf" in lp.names and "vcix.v.i" not in lp.names]
    assert not apart, [sorted(lp.names) for lp in apart]


def more_arrays_than_a_unit_has_are_refused():
    for raw, said in ((dict(_target.load(SYSTOLIC).as_dict(), matmul_units=2), "only the matrix unit"),
                      (dict(_target.load(MXU).as_dict(), matmul_units=0), "1 to 32"),
                      (dict(_target.load(MXU).as_dict(), matmul_units=33), "1 to 32")):
        try:
            _target.parse(raw, "<case>")
        except _target.TargetError as e:
            assert said in str(e), e
        else:
            raise AssertionError(f"matmul_units={raw['matmul_units']} was taken")


def the_pops_lag_the_pushes_by_the_delay_but_no_more_than_the_queues_hold():
    """At 128 lanes a word a cycle fills the 511-slot line in 64 inputs of 8 words; with queues
    of 1024 rows (64 bf16 inputs in, 64 out) the lag is the delay's, with 256 rows the queues'."""
    assert mxu_pass._lag(128, 8, 16, 1024) == 66
    assert mxu_pass._lag(128, 8, 8, 1024) == 66
    assert mxu_pass._lag(128, 8, 16, 256) == 32
    assert mxu_pass._lag(128, 8, 16, 16) == 2
    assert mxu_pass._lag(128, 8, 16, 8) == 1


CASES = [a_target_that_does_not_say_has_the_systolic_array,
         a_unit_no_lowering_knows_is_refused,
         a_bf16_dot_of_k_128_stages_the_first_half_and_latches_once,
         every_half_follows_its_vmatprep,
         a_panel_stages_its_first_half_then_its_second_and_the_next_vmatmul_latches,
         a_c_proven_zero_takes_the_first_pair_as_it_is,
         later_pairs_are_added_into_c_and_no_input_tests_its_turn,
         the_epilogue_after_a_dot_runs_at_the_last_pair_s_pops,
         the_dot_is_one_stream_across_its_n_pairs,
         the_model_is_given_no_delay_of_its_own,
         the_pops_lag_the_pushes_by_the_delay_but_no_more_than_the_queues_hold,
         two_arrays_take_the_k_pairs_by_turns_and_add_their_pops_in_registers,
         a_dot_of_one_k_tile_uses_the_first_array_alone,
         the_sample_gem5_times_drives_the_first_array_alone,
         a_dot_that_follows_one_with_the_same_weights_stages_none_again,
         a_k_loop_gives_its_first_and_last_trips_apart,
         a_k_loop_s_fill_goes_and_its_cast_runs_at_the_last_pops,
         more_arrays_than_a_unit_has_are_refused]

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
