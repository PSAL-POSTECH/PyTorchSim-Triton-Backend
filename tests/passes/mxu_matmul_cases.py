"""A target whose accelerator is vcix_tpuv6e lowers a dot for the matrix unit -- an operand as a vmatprep
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
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_matmul as mxu_pass  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

CASES_DIR = os.path.join(ROOT, "tests", "cases", "mechanism")
BF16 = os.path.join(CASES_DIR, "ops", "ops_bf16_matmul.py")          # 128 x 128 tile, K 128 a dot
WIDE = os.path.join(CASES_DIR, "tile", "tile_gemm_wide_tile_grid.py")  # f32, 256 x 256 tile, K 512
EPI = os.path.join(CASES_DIR, "tile", "tile_gemm_epilogue_at_pop.py")  # bf16 512^3, scale/relu/cast
MXU = _target.resolve("tpuv6e")
SHORT_F32 = os.path.join(CASES_DIR, "ops", "ops_f32_matmul_short_tiles.py")    # M 4, N 32, K 16
SHORT_BF16 = os.path.join(CASES_DIR, "ops", "ops_bf16_matmul_short_tiles.py")  # M 8, N 64, K 32
# bf16, two dots of 128 rows a window of B (K = 256, N = 128): one panel each, the same one
SHARED_B = os.path.join(CASES_DIR, "tile", "tile_gemm_two_dots_one_b.py")
SHARED_B_UNEQUAL = os.path.join(CASES_DIR, "tile", "tile_gemm_two_dots_one_b_unequal_m.py")
# bf16, K = 384 walked in blocks of 128 (acc += dot), N = 384 in a block of 512, cast after the loop
K_LOOP = os.path.join(CASES_DIR, "tile", "tile_gemm_k_loop_masked_n.py")


_Loop = collections.namedtuple("_Loop", "names stages")


def _vcix(o):
    # vcix.v.sv (the misc unit's pack) carries no immediate.
    imm = ir.IntegerAttr(o.attributes["imm"]).value if "imm" in o.attributes else None
    return (o.name, ir.IntegerAttr(o.attributes["opcode"]).value, imm)


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


def a_bf16_dot_of_k_128_stages_the_first_half_and_latches_once():
    dumps, ops, _, _, _, _ = _lowered(BF16, MXU)
    assert "01-tritonpytorchsim-lower-matmul-tpuv6e.mlir" in dumps, dumps
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


def _after_mxu(kernel):
    """Each vcix op of the matrix unit's dump as (name, opcode, the op defining its first operand's
    name), and every vector.transfer_write's vector type, walked from the dump right after the matrix unit's lowering."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                        "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                       env=dict(os.environ, TORCHSIM_COMPILE_TARGET=MXU,
                                TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        path = os.path.join(d, "09-passes", "01-tritonpytorchsim-lower-matmul-tpuv6e.mlir")
        with ir.Context() as ctx, open(path) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())
            ops = walk(module.operation)

            def source(o):
                owner = o.operands[0].owner
                return owner.name if isinstance(owner, (ir.Operation, ir.OpView)) else "block-arg"
            units = [(o.name, ir.IntegerAttr(o.attributes["opcode"]).value,
                      source(o) if len(o.operands) > 1 else None)
                     for o in ops if o.name.startswith("vcix.")]
            written = [ir.VectorType(o.operands[0].type) for o in ops if o.name == "vector.transfer_write"]
            writes = [(tuple(v.shape), v.element_type == ir.F32Type.get()) for v in written]
            names = collections.Counter(o.name for o in ops)
        return units, writes, names


def a_k_short_of_a_lane_tile_pushes_zero_weights_and_zeroes_a_s_lanes_past_it():
    """f32 K = 16 of 128: a panel is 16 pushes of 8 rows, of which the first two read B and the
    14 past K are the zero register; every input's lanes past K are selected to zero. The kernel
    walks K = 32 in two such blocks, a loop whose two trips come out of it: two dots."""
    units, _, names = _after_mxu(SHORT_F32)
    pushes = [src for name, opcode, src in units if name == "vcix.iv" and opcode == mxu_pass.VMATPUSH]
    dots = 2
    assert len(pushes) == 16 * dots, pushes
    assert pushes.count("arith.constant") == 14 * dots and \
        pushes.count("arith.bitcast") == 2 * dots, pushes
    inputs = [src for name, opcode, src in units if name == "vcix.iv" and opcode == mxu_pass.VMATMUL]
    assert inputs and all(src == "arith.select" for src in inputs), inputs
    assert names["torchsim.vlane_idx"] == dots, names


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
    pops = sum(1 for name, _, _ in units if name == "vcix.v.i")
    assert pops and pops == 4 * len(writes), (pops, writes)


def a_short_register_narrower_than_a_bank_moves_is_refused():
    """What the unit still refuses, with the bank's narrowest move: M = 1 or 2 of bf16 (a row that
    ends inside a word, a word alone), f32 M = 1, and a C that is not f32."""
    with ir.Context(), ir.Location.unknown():
        bf16, f32, f16 = ir.BF16Type.get(), ir.F32Type.get(), ir.F16Type.get()
        refused = [(1, 128, 16, 2, bf16, f32), (2, 128, 16, 2, bf16, f32),
                   (1, 128, 8, 1, f32, f32), (128, 128, 16, 2, f16, f16)]
        taken = [(8, 32, 16, 2, bf16, f32), (2, 16, 8, 1, f32, f32), (16, 8, 16, 2, bf16, f32)]
        for M, K, rows, pack, elt, acc in refused:
            try:
                mxu_pass._refuse("<dot>", M, K, 128, rows, pack, 512, elt, acc)
            except lib_matmul.ArrayRefusal:
                continue
            raise AssertionError(f"M={M} K={K} {elt} -> {acc} was taken")
        for M, K, rows, pack, elt, acc in taken:
            mxu_pass._refuse("<dot>", M, K, 128, rows, pack, 512, elt, acc)


def a_k_whose_last_pair_is_short_has_that_pair_apart():
    """K tiles in pairs: a short last tile is its pair's upper, or the odd tile alone."""
    assert mxu_pass._pairs(4, 128, 128) == [(0, 2, True, 128)]
    assert mxu_pass._pairs(4, 64, 128) == [(0, 1, True, 128), (2, 1, True, 64)]
    assert mxu_pass._pairs(2, 64, 128) == [(0, 1, True, 64)]
    assert mxu_pass._pairs(3, 64, 128) == [(0, 1, True, 128), (2, 1, False, 64)]
    assert mxu_pass._pairs(1, 16, 128) == [(0, 1, False, 16)]


def the_model_is_given_no_delay_of_its_own():
    """The v6e target states no delay, so the model takes its own; its queues are the target's,
    512 rows, and reach the model."""
    with tempfile.TemporaryDirectory() as d:
        with open(_target.load(MXU).write_model_yaml(os.path.join(d, "m.yml"))) as f:
            keys = yaml.safe_load(f)
        assert "tpu_mxu_delay_cycles" not in keys and keys["vpu_num_lanes"] == 128, keys
        assert keys["tpu_mxu_queue_entries"] == 512, keys


def the_pops_lag_the_pushes_by_the_delay_but_no_more_than_the_queues_hold():
    """At 128 lanes a word a cycle fills the 511-slot line in 64 inputs of 8 words; with queues
    of 1024 rows the lag is the delay's; the v6e target's 512 hold 64 bf16 inputs (f32: the delay's)."""
    assert mxu_pass._lag(128, 8, 16, 1024, None) == 66
    assert mxu_pass._lag(128, 8, 16, 512, None) == 64
    assert mxu_pass._lag(128, 8, 8, 512, None) == 66
    assert mxu_pass._lag(128, 8, 8, 1024, None) == 66
    assert mxu_pass._lag(128, 8, 16, 256, None) == 32
    assert mxu_pass._lag(128, 8, 16, 16, None) == 2
    assert mxu_pass._lag(128, 8, 16, 8, None) == 1


def a_stated_delay_is_the_target_s_and_an_unstated_queue_is_refused():
    """The lowering reads tpu_mxu_delay_cycles where the vcix block states it, and refuses, by the
    key's name, a target whose block does not state tpu_mxu_queue_entries; breaks if either is
    the lowering's own number again."""
    v6e = _target.load(MXU)
    said = dict(v6e.as_dict(), vcix={"tpu_mxu_queue_entries": 512, "tpu_mxu_delay_cycles": 63})
    assert mxu_pass._stated(_target.parse(said, "<delay 63>")) == (512, 63)
    assert mxu_pass._lag(128, 8, 8, 512, 63) == 10
    bare = _target.parse(dict(v6e.as_dict(), vcix={}), "<no queue>")
    with ir.Context(), ir.Location.unknown():
        try:
            mxu_pass.run(ir.Module.create(), target=bare)
            refused = None
        except _target.TargetError as e:
            refused = str(e)
    assert refused and "tpu_mxu_queue_entries" in refused, refused


def a_dot_that_follows_one_with_the_same_weights_stages_none_again():
    """Two dots in a block with one B, nothing writing it between, each one panel (K = 256 both
    halves, 16 pushes): they are one stream (_groups), so the panel is staged and latched once;
    the vmatmuls are the latching input, the first dot's other 7 and the second's 8."""
    _, ops, _, _, _, _ = _lowered(SHARED_B, MXU)
    count = collections.Counter((name, opcode) for name, opcode, _ in ops)
    assert count[("vcix.iv", mxu_pass.VMATPUSH)] == 16, count
    latched = [imm for name, opcode, imm in ops
               if name == "vcix.iv" and opcode == mxu_pass.VMATMUL and imm & mxu_pass.LATCH]
    assert len(latched) == 1, latched
    assert count[("vcix.iv", mxu_pass.VMATMUL)] == 3, count


def a_dot_on_the_same_weights_kept_apart_says_why():
    """Two dots on one B of 128 and 64 rows: their plans differ, so each is its own stream (two
    latches) and the second's is stamped KEPT_APART with that reason; two on one plan, none."""
    _, ops, _, _, kept, _ = _lowered(SHARED_B_UNEQUAL, MXU)
    latched = [imm for name, opcode, imm in ops
               if name == "vcix.iv" and opcode == mxu_pass.VMATMUL and imm & mxu_pass.LATCH]
    assert len(latched) == 2, latched
    assert kept == ["not one stream with the dot before: a different plan"], kept
    assert _lowered(SHARED_B, MXU)[4] == [], "two dots on one plan were kept apart"


def the_second_dot_of_a_window_follows_the_first_and_a_lone_dot_none():
    """_follows on the module before the lowering: the second dot of a window follows the first
    (same B, nothing writes it between); a kernel's lone dot follows nothing."""
    for kernel, want in ((SHARED_B, [False, True]), (EPI, [False])):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                            "--workdir", d, "--to", "transform"], check=True, capture_output=True,
                           env=dict(os.environ, TORCHSIM_COMPILE_TARGET=MXU,
                                    TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
            with ir.Context() as ctx, ir.Location.unknown(), \
                    open(os.path.join(d, "08-transformed.mlir")) as f:
                ctx.allow_unregistered_dialects = True
                module = ir.Module.parse(f.read())
                dots = [o for o in walk(module.operation) if o.name in lib_matmul.MATMUL_OPS]
                got = [bool(i) and mxu_pass._follows(dots[i - 1], o) for i, o in enumerate(dots)]
        assert got == want, (kernel, got)


def _facts_of(kernel):
    """The pass's step 2 on `kernel`'s 08-transformed.mlir, after step 1: each dot's _Facts, and
    every scf.for left, as (lower bound, upper bound) of constants."""
    from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import index_bank_casts
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import loop_bounds
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                        "--workdir", d, "--to", "transform"], check=True, capture_output=True,
                       env=dict(os.environ, TORCHSIM_COMPILE_TARGET=MXU,
                                TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        t = _target.load(MXU)
        with ir.Context() as ctx, ir.Location.unknown(), \
                open(os.path.join(d, "08-transformed.mlir")) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())
            index_bank_casts(module)
            mxu_pass._peel_k_loops(module)
            dots = [o for o in walk(module.operation) if o.name in lib_matmul.MATMUL_OPS]
            facts = [mxu_pass._facts(o, dots[i - 1] if i else None, t.vector_lanes, t.vlen_bits,
                                     (1, 1, 1)) for i, o in enumerate(dots)]
            loops = [loop_bounds(o)[:2] for o in walk(module.operation) if o.name == "scf.for"]
            return facts, loops


def _zero_weight_preps(kernel):
    """How many vmatpreps of weights (not of an input: no SECOND bit) carry the zero register for
    their upper columns, in the matrix unit's dump."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                        "--workdir", d, "--to", "lower"], check=True, capture_output=True,
                       env=dict(os.environ, TORCHSIM_COMPILE_TARGET=MXU,
                                TORCHSIM_COMPILE_IGNORE_EXPECT_IR="1"))
        path = os.path.join(d, "09-passes", "01-tritonpytorchsim-lower-matmul-tpuv6e.mlir")
        with ir.Context() as ctx, open(path) as f:
            ctx.allow_unregistered_dialects = True
            module = ir.Module.parse(f.read())
            from pytorchsim_triton_compiler.lowering.passes.lib_ir import producer
            n = 0
            for o in walk(module.operation):
                if (o.name == "vcix.iv"
                        and ir.IntegerAttr(o.attributes["opcode"]).value == mxu_pass.VMATPREP
                        and not ir.IntegerAttr(o.attributes["imm"]).value & mxu_pass.SECOND):
                    p = producer(o.operands[0])
                    n += p is not None and p.name == "arith.constant"
            return n


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
    """Lowered: no loop casts f32 to bf16 apart from the pops -- every loop that casts, by a truncf or
    the misc unit's pack (vcix.v.sv), also pops -- and three dots, each two panels of eight pushes (K one lane tile, the first half of the
    depth): N tiles 0 and 1 as a pair, and 2 alone, the zero register for its upper columns where
    lane tile 3, past N = 384, would be."""
    _, ops, loops, names, _, _ = _lowered(K_LOOP, MXU)
    casts = {"arith.truncf", "vcix.v.sv"}
    assert casts & names, names
    apart = [lp for lp in loops if casts & lp.names and "vcix.v.i" not in lp.names]
    assert not apart, [sorted(lp.names) for lp in apart]
    pushes = sum(1 for name, opcode, _ in ops if name == "vcix.iv" and opcode == mxu_pass.VMATPUSH)
    assert pushes == 3 * 2 * 8, pushes
    zero = _zero_weight_preps(K_LOOP)
    assert zero == 3 * 8, zero


CASES = [a_bf16_dot_of_k_128_stages_the_first_half_and_latches_once,
         every_half_follows_its_vmatprep,
         a_panel_stages_its_first_half_then_its_second_and_the_next_vmatmul_latches,
         a_c_proven_zero_takes_the_first_pair_as_it_is,
         later_pairs_are_added_into_c_and_no_input_tests_its_turn,
         the_epilogue_after_a_dot_runs_at_the_last_pair_s_pops,
         the_dot_is_one_stream_across_its_n_pairs,
         a_k_short_of_a_lane_tile_pushes_zero_weights_and_zeroes_a_s_lanes_past_it,
         an_m_short_of_a_register_pushes_zero_words_and_writes_only_its_rows,
         the_rows_past_m_are_popped_and_dropped,
         a_short_register_narrower_than_a_bank_moves_is_refused,
         a_k_whose_last_pair_is_short_has_that_pair_apart,
         the_model_is_given_no_delay_of_its_own,
         the_pops_lag_the_pushes_by_the_delay_but_no_more_than_the_queues_hold,
         a_stated_delay_is_the_target_s_and_an_unstated_queue_is_refused,
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
