"""A dot lowers at the matrix unit's shape: an operand of two registers on the v6e, one on the v3.
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
from pytorchsim_triton_compiler.targets.tpu.pytorchsim_tpu import (
    MiscPackOp, MxuMatmulOp, MxuPopOp, MxuPrepOp, MxuPushOp)
from pytorchsim_triton_compiler.passes.utils import matmul  # noqa: E402
from pytorchsim_triton_compiler.targets.tpu import lower_matmul as mxu_pass  # noqa: E402
from pytorchsim_triton_compiler.passes.conversion import to_vcix  # noqa: E402
from pytorchsim_triton_compiler.passes.utils.ir_utils import walk  # noqa: E402

CASES_DIR = os.path.join(ROOT, "tests", "cases", "mechanism")
BF16 = os.path.join(CASES_DIR, "ops", "ops_bf16_matmul.py")          # 128 x 128 tile, K 128 a dot
WIDE = os.path.join(CASES_DIR, "tile", "tile_gemm_wide_tile_grid.py")  # f32, 256 x 256 tile, K 512
EPI = os.path.join(CASES_DIR, "tile", "tile_gemm_epilogue_at_pop.py")  # bf16 512^3, scale/relu/cast
MXU = _target.resolve("tpuv6e")
V3 = _target.resolve("tpuv3")
V6E_SHAPE, V3_SHAPE = mxu_pass.Shape(2, True), mxu_pass.Shape(1, False)
SHORT_F32 = os.path.join(CASES_DIR, "ops", "ops_f32_matmul_short_tiles.py")    # M 4, N 32, K 16
SHORT_BF16 = os.path.join(CASES_DIR, "ops", "ops_bf16_matmul_short_tiles.py")  # M 8, N 64, K 32
# bf16, two dots of 128 rows a window of B (K = 256, N = 128): one panel each, the same one
SHARED_B = os.path.join(CASES_DIR, "tile", "tile_gemm_two_dots_one_b.py")
SHARED_B_UNEQUAL = os.path.join(CASES_DIR, "tile", "tile_gemm_two_dots_one_b_unequal_m.py")
# bf16, K = 384 walked in blocks of 128 (acc += dot), N = 384 in a block of 512, cast after the loop
K_LOOP = os.path.join(CASES_DIR, "tile", "tile_gemm_k_loop_masked_n.py")
FEW = os.path.join(CASES_DIR, "tile", "tile_gemm_streamed_few_inputs.py")  # bf16 M 32, K 2048
# Causal flash attention: QK^T's mask and row max walk at its pops, then P V
ROW_FOLD = os.path.join(CASES_DIR, "loops", "loop_08_trip_from_program_id_tog.py")


_Loop = collections.namedtuple("_Loop", "names stages")
PREP, PUSH, MATMUL, POP = (MxuPrepOp.OPERATION_NAME, MxuPushOp.OPERATION_NAME, MxuMatmulOp.OPERATION_NAME,
                           MxuPopOp.OPERATION_NAME)
MXU_OPS = (PREP, PUSH, MATMUL, POP)
_FLAG = {PREP: "input", PUSH: "second_half", MATMUL: "latch", POP: "upper"}


def _unit(o):
    """Matrix unit op `o` as (name, its format or None for a pop, its flag: input, second_half,
    latch or upper)."""
    fmt = ir.IntegerAttr(o.attributes["format"]).value if "format" in o.attributes else None
    return (o.name, fmt, ir.BoolAttr(o.attributes[_FLAG[o.name]]).value)


def _stages(o):
    """A vmatpush: a half of the weights staged."""
    return o.name == PUSH


def _units_dump(d):
    """The 09-passes dump convert-to-vcix reads: the last that holds the unit ops, not yet VCIX."""
    dumps = sorted(os.listdir(os.path.join(d, "09-passes")))
    at = [n.split("-", 1)[1] for n in dumps].index(f"{to_vcix.NAME}.mlir")
    assert at, dumps
    return os.path.join(d, "09-passes", dumps[at - 1])


def _lowered(kernel, subtarget):
    """_units_dump walked: (the pass dumps' names, every matrix unit op as _unit gives it, each
    affine.for as a _Loop, every op name in the module, what each op stamped KEPT_APART says, how
    many pops sit in no loop that also stages weights). The kernel's expect_ir is not read."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), kernel,
                        "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                       env=dict(os.environ, PYTORCHSIM_COMPILE_TARGET=subtarget,
                                PYTORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        found, loops, free = [], [], [0]
        with ir.Context() as ctx, open(_units_dump(d)) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())
            everything = walk(module.operation)

            def visit(op, under_weights):
                for region in op.regions:
                    for block in region.blocks:
                        for inner in block.operations:
                            o = inner.operation
                            if o.name in MXU_OPS:
                                found.append(_unit(o))
                                free[0] += o.name == POP and not under_weights
                            if o.name == "affine.for":
                                inside = walk(o)
                                loops.append(_Loop({i.name for i in inside},
                                                   any(_stages(i) for i in inside)))
                                visit(o, under_weights or loops[-1].stages)
                            else:
                                visit(o, under_weights)
            visit(module.operation, False)
            names = {o.name for o in everything}
            kept = [ir.StringAttr(o.attributes[matmul.KEPT_APART]).value
                    for o in everything if matmul.KEPT_APART in o.attributes]
        return sorted(os.listdir(os.path.join(d, "09-passes"))), found, loops, names, kept, free[0]


def a_bf16_dot_of_k_128_stages_the_first_half_and_latches_once():
    """K = 128 a block, eight registers a group: one pair into the first half of the depth (format 3). K walks
    two blocks, peeled into two dots (_peel_k_loops); M = 128 is one input a dot, which latches.

    Its 128 rows are two pops of 64 a half of the columns."""
    dumps, ops, _, names, _, _ = _lowered(BF16, MXU)
    assert "01-convert-pytorchsim-to-pytorchsim-tpu.mlir" in dumps, dumps
    count = collections.Counter(ops)
    dots, bf16 = 2, mxu_pass.FORMAT_BF16
    assert count[(PREP, bf16, False)] == dots and count[(PUSH, bf16, False)] == dots, count
    assert not any(name == PUSH and second for name, _, second in ops), count
    assert count[(PREP, bf16, True)] == dots, count
    assert count[(MATMUL, bf16, True)] == dots and count[(MATMUL, bf16, False)] == 0, count
    assert count[(POP, None, False)] == 2 * dots and count[(POP, None, True)] == 2 * dots, count
    assert "vcix.i" not in names, "a compute marker on the matrix unit"


def an_operand_of_one_register_takes_no_vmatprep_and_no_upper_half():
    """On pytorchsim_tpuv3 a dot has no vmatprep, second half or upper pop (tpu/mxu.hpp refuses them)."""
    _, ops, _, _, _, _ = _lowered(BF16, V3)
    count = collections.Counter(ops)
    dots, bf16 = 2, mxu_pass.FORMAT_BF16
    assert count[(PUSH, bf16, False)] == dots and count[(MATMUL, bf16, True)] == dots, count
    assert count[(POP, None, False)] == 2 * dots, count
    for kernel in (BF16, WIDE):
        _, unit, _, _, _, _ = _lowered(kernel, V3)
        assert unit and not any(name == PREP for name, _, _ in unit), unit
        assert all(not second for name, _, second in unit if name == PUSH), unit
        assert all(not upper for name, _, upper in unit if name == POP), unit


def every_half_follows_its_vmatprep():
    _, ops, _, _, _, _ = _lowered(WIDE, MXU)
    pushes = [(name, fmt, flag) for name, fmt, flag in ops if name != POP]
    assert pushes and len(pushes) % 2 == 0, len(pushes)
    for (prep, prep_fmt, of_input), (half, half_fmt, _) in zip(pushes[0::2], pushes[1::2]):
        assert prep == PREP and half == (MATMUL if of_input else PUSH), (prep, of_input, half)
        assert prep_fmt == half_fmt == mxu_pass.FORMAT_F32, "f32 words are format 0"


def _run_order(op, out):
    """The ops under `op` in the order they run: a loop of constant trips its body that many
    times; any other op with regions its regions once."""
    from pytorchsim_triton_compiler.passes.utils.ir_utils import trip_count
    for region in op.regions:
        for block in region.blocks:
            for o in [x.operation for x in block.operations]:
                out.append(o)
                if o.name in ("affine.for", "scf.for"):
                    trips = trip_count(o)
                    assert trips is not None, f"a loop of unknown trips: {o.name}"
                    for _ in range(trips):
                        _run_order(o, out)
                else:
                    _run_order(o, out)
    return out


def _ran(kernel, dump=None):
    """Every matrix unit op of `kernel`'s `dump` (one of 09-passes; _units_dump's if None) in the order
    the program issues it, as _unit gives it with the name of the op defining its first operand."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), kernel,
                        "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                       env=dict(os.environ, PYTORCHSIM_COMPILE_TARGET=MXU,
                                PYTORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        path = _units_dump(d) if dump is None else os.path.join(d, "09-passes", dump)
        with ir.Context() as ctx, open(path) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())

            def source(o):
                owner = o.operands[0].owner
                return owner.name if isinstance(owner, (ir.Operation, ir.OpView)) else "block-arg"
            return [_unit(o) + (source(o) if len(o.operands) > 1 else None,)
                    for o in _run_order(module.operation, []) if o.name in MXU_OPS]


def a_panel_stages_its_first_half_then_its_second_and_the_next_vmatmul_latches():
    """f32, K = 512: two K pairs with an upper tile, a half of a lane tile's rows pushed in as many
    pushes as the dot's group holds rows (_push_rows). In the order the
    program runs, the pushes before each latching vmatmul since the one before are a panel whole,
    its first half then its second, and only a panel's first vmatmul latches."""
    group, = _widths(WIDE)
    per = _target.load(MXU).vector_lanes // _push_rows(group, 1)
    pushes, latched, computed = [], 0, 0
    f32 = mxu_pass.FORMAT_F32
    for name, fmt, flag, _ in _ran(WIDE):
        if name == PUSH:
            pushes.append((fmt, flag))
        elif name == MATMUL and flag:
            assert pushes == [(f32, False)] * per + [(f32, True)] * per, (group, pushes)
            pushes, latched = [], latched + 1
        elif name == MATMUL:
            computed += 1
    assert latched >= 2 and computed >= 1 and not pushes, (latched, computed, pushes)


def the_next_panel_is_staged_among_the_inputs_of_the_one_in_use():
    """f32, K = 512, M = 256, 64 rows a group: 4 inputs a panel. The second panel's 4 pushes run
    between the first panel's vmatmuls, as the v6e pushes the next tile under the one in use: at
    most one push after each vmatmul, the last after the first panel's last input. Breaks if the
    panel is staged whole after the inputs before it."""
    ops = [(name, flag) for name, _, flag, _ in _ran(WIDE) if name != POP]
    latches = [k for k, (name, latch) in enumerate(ops) if name == MATMUL and latch]
    assert len(latches) == 2, latches
    between, most = 0, 0
    for name, _ in ops[latches[0] + 1:latches[1]]:
        if name == PUSH:
            between += 1
        elif name == MATMUL:
            most, between = max(most, between), 0
    assert most <= 1 and between <= 1, (most, between)


def a_panel_with_fewer_inputs_than_pairs_is_pushed_once():
    """bf16, M = 32, K = 2048: a group of two registers (32 rows, M's), one input a panel, eight
    panels of 8 pairs. As the program runs, every panel is 8 pushes, at most one after each input
    that latches nothing, the rest before the next latch. Breaks if a looped K pair pushes the panel
    the pair before already staged."""
    ops = [(name, flag) for name, _, flag, _ in _ran(FEW) if name != POP]
    latches = [k for k, (name, latch) in enumerate(ops) if name == MATMUL and latch]
    assert len(latches) == 8, latches
    pushes = [sum(1 for name, _ in ops[a:b] if name == PUSH)
              for a, b in zip([0] + latches, latches)]
    assert pushes == [8] * 8, pushes
    for a, b in zip(latches, latches[1:]):
        inputs = [k for k in range(a, b) if ops[k][0] == MATMUL]
        for k, nxt in zip(inputs, inputs[1:]):
            between = sum(1 for name, _ in ops[k:nxt] if name == PUSH)
            assert between <= 1, (a, k, between)


def a_c_proven_zero_takes_the_first_pair_as_it_is():
    """f32, K = 512, two K pairs, C filled with zeros before the dot: the first pair's pops are
    written as they are, the second's added. (That the fill goes with its loop is held by the
    epilogue case below; this kernel gathers A in loops of its own.)"""
    _, _, loops, _, _, _ = _lowered(WIDE, MXU)
    popping = [loop.names for loop in loops if POP in loop.names]
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
    assert any({"arith.truncf", POP} <= loop.names for loop in loops), "no cast at a pop"
    assert loops and all(any(n in MXU_OPS for n in loop.names) for loop in loops), \
        "the epilogue stays a loop"


def the_dot_is_one_stream_across_its_n_pairs():
    """bf16 512^3: two N pairs, two K pairs. Were each N pair a stream of its own, every pop would
    sit in the N pairs' loop beside the weights it stages; in one stream the dot's drain comes
    after the last pair, in loops that stage nothing."""
    _, _, _, _, _, free_pops = _lowered(EPI, MXU)
    assert free_pops > 0, "every pop is inside a loop that also stages weights: a stream per N pair"


def _after_mxu(kernel):
    """Each matrix unit op as (name, the op defining its first operand's name), and every
    vector.transfer_write's vector type, walked from the dump right after the matrix unit's lowering."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), kernel,
                        "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                       env=dict(os.environ, PYTORCHSIM_COMPILE_TARGET=MXU,
                                PYTORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        path = os.path.join(d, "09-passes", "01-convert-pytorchsim-to-pytorchsim-tpu.mlir")
        with ir.Context() as ctx, open(path) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())
            ops = walk(module.operation)

            def source(o):
                owner = o.operands[0].owner
                return owner.name if isinstance(owner, (ir.Operation, ir.OpView)) else "block-arg"
            units = [(o.name, source(o) if len(o.operands) > 1 else None) for o in ops if o.name in MXU_OPS]
            written = [ir.VectorType(o.operands[0].type) for o in ops if o.name == "vector.transfer_write"]
            writes = [(tuple(v.shape), v.element_type == ir.F32Type.get()) for v in written]
            names = collections.Counter(o.name for o in ops)
        return units, writes, names


def a_k_short_of_a_lane_tile_pushes_zero_weights_and_zeroes_a_s_lanes_past_it():
    """f32 K = 16 of 128: a panel is 16 pushes of 8 rows, the first two reading B and the 14 past K the
    zero register; every input's lanes past K are selected to zero (a select or its bitwise form).

    K = 32 walks two such blocks, a loop whose two trips come out of it: two dots."""
    units, _, names = _after_mxu(SHORT_F32)
    pushes = [src for name, src in units if name == PUSH]
    dots = 2
    assert len(pushes) == 16 * dots, pushes
    assert pushes.count("arith.constant") == 14 * dots and \
        pushes.count("arith.bitcast") == 2 * dots, pushes
    inputs = [src for name, src in units if name == MATMUL]
    assert inputs and all(src in ("arith.select", "arith.xori") for src in inputs), inputs
    assert names["pytorchsim_tpu.misc.lane_idx"] == dots, names


def an_m_short_of_a_register_pushes_zero_words_and_writes_only_its_rows():
    """f32 M = 4 of a register's 8: the input reads four words into a zero register, and every
    write of a pop into C is the four rows."""
    _, writes, names = _after_mxu(SHORT_F32)
    assert names["vector.insert_strided_slice"] >= 1, names
    assert writes and all(w == ((4,), True) for w in writes), writes


def the_rows_past_m_are_popped_and_dropped():
    """bf16 M = 8 of a register's 16, N = 64 of a lane tile: each input is four pops (two registers
    of f32 rows, lower and upper columns) and one of them -- rows 0..7, lower columns -- is
    written."""
    units, writes, _ = _after_mxu(SHORT_BF16)
    pops = sum(1 for name, _ in units if name == POP)
    assert pops and pops == 4 * len(writes), (pops, writes)


def one_row_is_refused_and_a_one_word_tail_is_taken():
    """M = 1 is a vector op and the unit refuses it, as a row ending inside a word (bf16 M = 1)
    and a C that is not f32; a last register of one word (f32 M = 129, bf16 M = 2) is taken."""
    with ir.Context(), ir.Location.unknown():
        bf16, f32, f16 = ir.BF16Type.get(), ir.F32Type.get(), ir.F16Type.get()
        refused = [(1, 128, 16, 2, bf16, f32), (1, 128, 8, 1, f32, f32), (1, 16, 8, 1, f32, f32),
                   (128, 128, 16, 2, f16, f16)]
        taken = [(8, 32, 16, 2, bf16, f32), (2, 16, 8, 1, f32, f32), (16, 8, 16, 2, bf16, f32),
                 (2, 128, 16, 2, bf16, f32), (129, 128, 8, 1, f32, f32), (8, 129, 8, 1, f32, f32)]
        for M, K, rows, pack, elt, acc in refused:
            try:
                mxu_pass._refuse("<dot>", M, K, rows, pack, 512, elt, acc)
            except matmul.ArrayRefusal:
                continue
            raise AssertionError(f"M={M} K={K} {elt} -> {acc} was taken")
        for M, K, rows, pack, elt, acc in taken:
            mxu_pass._refuse("<dot>", M, K, rows, pack, 512, elt, acc)


def a_k_whose_last_pair_is_short_has_that_pair_apart():
    """K tiles in pairs: a short last tile is its pair's upper, or the odd tile alone."""
    assert mxu_pass._pairs(4, 128, 128, 2) == [(0, 2, True, 128)]
    assert mxu_pass._pairs(4, 64, 128, 2) == [(0, 1, True, 128), (2, 1, True, 64)]
    assert mxu_pass._pairs(2, 64, 128, 2) == [(0, 1, True, 64)]
    assert mxu_pass._pairs(3, 64, 128, 2) == [(0, 1, True, 128), (2, 1, False, 64)]
    assert mxu_pass._pairs(1, 16, 128, 2) == [(0, 1, False, 16)]
    assert mxu_pass._pairs(4, 128, 128, 1) == [(0, 4, False, 128)]
    assert mxu_pass._pairs(4, 64, 128, 1) == [(0, 3, False, 128), (3, 1, False, 64)]


def the_pops_lag_the_pushes_by_the_delay_but_no_more_than_the_queues_hold():
    """At 128 lanes a word a cycle fills the 511-slot line in 64 inputs of 8 words; queues of 1024 rows lag
    by the delay, the v6e's 512 hold 64 bf16 inputs; the v3's line is 255 slots, 16 cycles an input."""
    assert mxu_pass._lag(128, 8, 16, 1024, None, V6E_SHAPE, 1) == 66
    assert mxu_pass._lag(128, 8, 16, 512, None, V6E_SHAPE, 1) == 64
    assert mxu_pass._lag(128, 8, 8, 512, None, V6E_SHAPE, 1) == 66
    assert mxu_pass._lag(128, 8, 8, 1024, None, V6E_SHAPE, 1) == 66
    assert mxu_pass._lag(128, 8, 16, 256, None, V6E_SHAPE, 1) == 32
    assert mxu_pass._lag(128, 8, 16, 16, None, V6E_SHAPE, 1) == 2
    assert mxu_pass._lag(128, 8, 16, 8, None, V6E_SHAPE, 1) == 1
    assert mxu_pass._lag(128, 8, 16, 512, None, V3_SHAPE, 1) == 18
    assert mxu_pass._lag(128, 8, 8, 512, 97, V3_SHAPE, 1) == 15
    assert mxu_pass._lag(128, 8, 16, 256, None, V3_SHAPE, 1) == 16


def the_pops_trail_a_whole_panel_of_inputs_when_the_queues_hold_it():
    """A panel of more inputs than the delay asks: they all go in before the first pop, as far as the
    queues hold them (v6e, a stated delay of 187 cycles, 16 words an input: 14 by the delay, 32 held)."""
    assert mxu_pass._lag(128, 16, 32, 512, 187, V6E_SHAPE, 1) == 14
    assert mxu_pass._lag(128, 16, 32, 512, 187, V6E_SHAPE, 20) == 20
    assert mxu_pass._lag(128, 16, 32, 512, 187, V6E_SHAPE, 32) == 32
    assert mxu_pass._lag(128, 16, 32, 512, 187, V6E_SHAPE, 64) == 32


def a_stated_delay_is_the_target_s_and_an_unstated_queue_is_refused():
    """The lowering reads tpu_mxu_delay_cycles where the vcix block states it, and refuses, by the
    key's name, a subtarget whose block does not state tpu_mxu_queue_entries; breaks if either is
    the lowering's own number again."""
    v6e = _target.load(MXU)
    said = dict(v6e.as_dict(), vcix={"tpu_mxu_queue_entries": 512, "tpu_mxu_delay_cycles": 63})
    assert mxu_pass._stated(_target.parse(said, "<delay 63>")) == (512, 63)
    assert mxu_pass._lag(128, 8, 8, 512, 63, V6E_SHAPE, 1) == 10
    bare = _target.parse(dict(v6e.as_dict(), vcix={}), "<no queue>")
    with ir.Context(), ir.Location.unknown():
        try:
            mxu_pass.run(ir.Module.create(), subtarget=bare)
            refused = None
        except _target.TargetError as e:
            refused = str(e)
    assert refused and "tpu_mxu_queue_entries" in refused, refused


def the_shape_is_the_machine_file_s_and_an_unstated_one_is_refused():
    """The stream follows the machine file's tpu_mxu_registers/word_a_cycle; none, or R 3, is refused."""
    assert mxu_pass._shape(_target.load(MXU)) == V6E_SHAPE and mxu_pass._shape(_target.load(V3)) == V3_SHAPE
    v6e = _target.load(MXU)
    said = {}
    for name, vcix in (("none", {"tpu_mxu_queue_entries": 512}),
                       ("R 3", dict(v6e.vcix, tpu_mxu_registers=3))):
        try:
            mxu_pass._shape(_target.parse(dict(v6e.as_dict(), vcix=vcix), f"<{name}>"))
            said[name] = None
        except _target.TargetError as e:
            said[name] = str(e)
    assert said["none"] and "tpu_mxu_registers" in said["none"] and "tpu_mxu_word_a_cycle" in said["none"], said
    assert said["R 3"] and "tpu_mxu_registers is 3" in said["R 3"], said


def a_dot_that_follows_one_with_the_same_weights_stages_none_again():
    """Two dots in a block with one B, nothing writing it between, each one panel (K = 256 both
    halves, as many pushes as the dots' group holds rows): they are one stream (_groups), so the
    panel is staged and latched once, by the first dot's first input."""
    group = set(_widths(SHARED_B))
    assert len(group) == 1, group
    _, ops, _, _, _, _ = _lowered(SHARED_B, MXU)
    count = collections.Counter(name for name, _, _ in ops)
    pushes = 256 // _push_rows(group.pop(), 2)
    assert count[PUSH] == pushes, (pushes, count)
    latched = [fmt for name, fmt, latch in ops if name == MATMUL and latch]
    assert len(latched) == 1, latched


def a_dot_on_the_same_weights_kept_apart_says_why():
    """Two dots on one B of 128 and 64 rows: their plans differ, so each is its own stream (two
    latches) and the second's is stamped KEPT_APART with that reason; two on one plan, none."""
    _, ops, _, _, kept, _ = _lowered(SHARED_B_UNEQUAL, MXU)
    latched = [fmt for name, fmt, latch in ops if name == MATMUL and latch]
    assert len(latched) == 2, latched
    assert kept == ["not one stream with the dot before: a different plan"], kept
    assert _lowered(SHARED_B, MXU)[4] == [], "two dots on one plan were kept apart"


def the_second_dot_of_a_window_follows_the_first_and_a_lone_dot_none():
    """_follows on the module before the lowering: the second dot of a window follows the first
    (same B, nothing writes it between); a kernel's lone dot follows nothing."""
    for kernel, want in ((SHARED_B, [False, True]), (EPI, [False])):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), kernel,
                            "--workdir", d, "--to", "transform"], check=True, capture_output=True,
                           env=dict(os.environ, PYTORCHSIM_COMPILE_TARGET=MXU,
                                    PYTORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
            with ir.Context() as ctx, ir.Location.unknown(), \
                    open(os.path.join(d, "08-transformed.mlir")) as f:
                ctx.allow_unregistered_dialects = True
                module = ir.Module.parse(f.read())
                dots = [o for o in walk(module.operation) if o.name in matmul.MATMUL_OPS]
                got = [bool(i) and mxu_pass._follows(dots[i - 1], o) for i, o in enumerate(dots)]
        assert got == want, (kernel, got)


def _facts_of(kernel):
    """The pass's step 2 on `kernel`'s 08-transformed.mlir, after step 1: each dot's _Facts, and
    every scf.for left, as (lower bound, upper bound) of constants."""
    from pytorchsim_triton_compiler.passes.utils.ir_utils import loop_bounds
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), kernel,
                        "--workdir", d, "--to", "transform"], check=True, capture_output=True,
                       env=dict(os.environ, PYTORCHSIM_COMPILE_TARGET=MXU,
                                PYTORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        t = _target.load(MXU)
        with ir.Context() as ctx, ir.Location.unknown(), \
                open(os.path.join(d, "08-transformed.mlir")) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())
            mxu_pass._peel_k_loops(module)
            dots = [o for o in walk(module.operation) if o.name in matmul.MATMUL_OPS]
            facts = [mxu_pass._facts(o, dots[i - 1] if i else None, t.vector_lanes, t.vlen_bits,
                                     (1, 1, 1)) for i, o in enumerate(dots)]
            loops = [loop_bounds(o)[:2] for o in walk(module.operation) if o.name == "scf.for"]
            return facts, loops


def a_k_loop_gives_its_first_and_last_trips_apart():
    """K = 384 in blocks of 128 is three trips: the first and the last come out of the loop, the
    middle one stays. Only the first takes a C its fill proves zero; each trip's K block is one
    lane tile, and N = 384 is three of the block's four lane tiles in every trip."""
    facts, loops = _facts_of(K_LOOP)
    assert len(facts) == 3 and len(loops) == 1, (facts, loops)
    # The middle trips' loop starts at 0: the TOG's sample runs a loop's trip from its start, and
    # one that started past it ran none, so its markers never fired.
    assert loops == [(0, 128)], loops
    assert [f.init is not None for f in facts] == [True, False, False], facts
    assert [f.live for f in facts] == [(3, 1)] * 3, facts


def a_k_loop_s_fill_goes_and_its_cast_runs_at_the_last_pops():
    """No loop casts f32 to bf16 apart from the pops: every loop that casts (truncf or misc.pack) pops.

    Three dots, each two panels of two pushes (K one lane tile, first half of the depth; M = 64): N tiles
    0 and 1 a pair, 2 alone with the zero register for the upper columns where tile 3 (past N) would be."""
    _, _, loops, names, _, _ = _lowered(K_LOOP, MXU)
    casts = {"arith.truncf", MiscPackOp.OPERATION_NAME}
    assert casts & names, names
    apart = [lp for lp in loops if casts & lp.names and POP not in lp.names]
    assert not apart, [sorted(lp.names) for lp in apart]
    ran = _ran(K_LOOP, "01-convert-pytorchsim-to-pytorchsim-tpu.mlir")
    pushes = sum(1 for name, _, _, _ in ran if name == PUSH)
    assert pushes == 3 * 2 * 2, pushes
    zero = sum(1 for name, _, of_input, src in ran
               if name == PREP and not of_input and src == "arith.constant")
    assert zero == 3 * 2, zero


def _push_rows(group, pack):
    """The rows one push of a group of `group` registers stages, `pack` rows to a 32-bit word."""
    return _target.load(MXU).vlen_bits * group // 32 * pack


def _widths(kernel, **opts):
    """The LMUL each dot of `kernel` is lowered at (the pass's dot_widths), in program order, the
    pass run over its 08-transformed.mlir with `opts`."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), kernel,
                        "--workdir", d, "--to", "transform"], check=True, capture_output=True,
                       env=dict(os.environ, PYTORCHSIM_COMPILE_TARGET=MXU,
                                PYTORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        with ir.Context() as ctx, ir.Location.unknown(), \
                open(os.path.join(d, "08-transformed.mlir")) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())
            got = {}
            mxu_pass.run(module, subtarget=_target.load(MXU), dot_widths=got, **opts)
            return [got[n] for n in sorted(got)]


def a_dot_s_group_is_the_widest_the_unit_its_m_and_its_epilogue_allow():
    """A push, an input and a pop take a group of registers: bf16 M = K = 128 eight (128 rows, half
    the depth); M = 32 two (a wider group's rows past M are zeros); f32 M = 4 one. Attention's QK^T
    two: at four its stream's groups and what the mask walk at its pops holds (a row's lane index,
    64 bits an element, and its running max) pass the vector registers; its P V, whose epilogue is
    not run at the pops, eight."""
    for kernel, want in ((BF16, [8, 8]), (FEW, [2]), (SHORT_F32, [1, 1]), (ROW_FOLD, [2, 8])):
        got = _widths(kernel)
        assert got == want, (os.path.basename(kernel), got, want)


def the_retry_narrows_the_dot_it_names_and_the_kernel_s_lmul_every_dot():
    """The retry's dot_lmul narrows that dot alone; the kernel's LMUL bounds them all."""
    assert _widths(BF16, dot_lmul={1: 2}) == [2, 8]
    assert _widths(BF16, lmul=4) == [4, 4]


CASES = [a_dot_s_group_is_the_widest_the_unit_its_m_and_its_epilogue_allow,
         the_retry_narrows_the_dot_it_names_and_the_kernel_s_lmul_every_dot,
         a_panel_with_fewer_inputs_than_pairs_is_pushed_once,
         a_bf16_dot_of_k_128_stages_the_first_half_and_latches_once,
         every_half_follows_its_vmatprep,
         an_operand_of_one_register_takes_no_vmatprep_and_no_upper_half,
         a_panel_stages_its_first_half_then_its_second_and_the_next_vmatmul_latches,
         the_next_panel_is_staged_among_the_inputs_of_the_one_in_use,
         a_c_proven_zero_takes_the_first_pair_as_it_is,
         later_pairs_are_added_into_c_and_no_input_tests_its_turn,
         the_epilogue_after_a_dot_runs_at_the_last_pair_s_pops,
         the_dot_is_one_stream_across_its_n_pairs,
         a_k_short_of_a_lane_tile_pushes_zero_weights_and_zeroes_a_s_lanes_past_it,
         an_m_short_of_a_register_pushes_zero_words_and_writes_only_its_rows,
         the_rows_past_m_are_popped_and_dropped,
         one_row_is_refused_and_a_one_word_tail_is_taken,
         a_k_whose_last_pair_is_short_has_that_pair_apart,
         the_pops_lag_the_pushes_by_the_delay_but_no_more_than_the_queues_hold,
         the_pops_trail_a_whole_panel_of_inputs_when_the_queues_hold_it,
         a_stated_delay_is_the_target_s_and_an_unstated_queue_is_refused,
         the_shape_is_the_machine_file_s_and_an_unstated_one_is_refused,
         a_dot_that_follows_one_with_the_same_weights_stages_none_again,
         a_dot_on_the_same_weights_kept_apart_says_why,
         the_second_dot_of_a_window_follows_the_first_and_a_lone_dot_none,
         a_k_loop_gives_its_first_and_last_trips_apart,
         a_k_loop_s_fill_goes_and_its_cast_runs_at_the_last_pops]

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
