"""lower_cross_lane emits a permute's rows as the unit takes them: its bank
reads and writes are in bounds, read off the IR right after the pass.
    python3 tests/passes/xlu_permute_emission_cases.py   exit 0 iff all hold"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lib_xlu  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family.lower_cross_lane import (  # noqa: E402
    _SF_VC_IVV_SE)
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

SPLIT = os.path.join(ROOT, "tests", "cases", "mechanism", "ptr", "ptr_div_only_axis_split.py")
CUMSUM = os.path.join(ROOT, "tests", "cases", "mechanism", "ops", "ops_cumsum_bool_promoted.py")
PERMUTE_I64 = os.path.join(ROOT, "tests", "cases", "mechanism", "ops", "ops_xlu_permute_i64.py")
TRANSFERS = ("vector.transfer_read", "vector.transfer_write")
_DUMPS = {}


def _after_cross_lane(kernel):
    """The text of the IR right after the cross-lane pass, compiled once per kernel."""
    if kernel not in _DUMPS:
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "w")
            subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                            "--workdir", w, "--to", "lower"], check=False, capture_output=True)
            out = [f for f in sorted(os.listdir(os.path.join(w, "09-passes")))
                   if "lower-cross-lane" in f]
            _DUMPS[kernel] = open(os.path.join(w, "09-passes", out[0])).read()
    return _DUMPS[kernel]


def _is_push(o):
    return (o.name == "vcix.iv"
            and ir.IntegerAttr(o.attributes["opcode"]).value == lib_xlu.XLU_PUSH_OPCODE)


def _is_set(o):
    if o.name != "llvm.call_intrinsic" or ir.StringAttr(o.attributes["intrin"]).value != _SF_VC_IVV_SE:
        return False
    code = o.operands[0].owner
    return ir.IntegerAttr(code.attributes["value"]).value == lib_xlu.XLU_SET_PATTERN_OPCODE


def _is_pop(o):
    return (o.name == "vcix.v.i"
            and ir.IntegerAttr(o.attributes["opcode"]).value == lib_xlu.XLU_POP_OPCODE)


def _unit_blocks(m):
    """Each op that directly holds an XLU push or pop: (the ops beside them, whether it pops)."""
    held = []
    for o in walk(m.operation):
        if _is_push(o) or _is_pop(o):
            p = o.parent
            for i, (q, ops, pops) in enumerate(held):
                if q == p:
                    held[i] = (q, ops, pops or _is_pop(o))
                    break
            else:
                held.append((p, [x for r in p.regions for b in r.blocks for x in b.operations],
                             _is_pop(o)))
    return [(ops, pops) for _, ops, pops in held]


def _in_bounds(o):
    return [ir.BoolAttr(x).value for x in ir.ArrayAttr(o.attributes["in_bounds"])]


def _masked(o):
    return list(ir.DenseI32ArrayAttr(o.attributes["operandSegmentSizes"]))[-1] > 0


def _store_of(pop):
    """The transfer_write a pop's value reaches, and the names of the ops that feed that store."""
    todo, seen = [pop], []
    while todo:
        o = todo.pop()
        for r in o.results:
            for u in r.uses:
                x = u.owner
                if x.name == TRANSFERS[1]:
                    return x, _feeders(x)
                if not any(x == y for y in seen):
                    seen.append(x)
                    todo.append(x)
    return None, []


def _feeders(store):
    """The names of the ops the stored value is computed from, back to the pop."""
    names, todo = [], [store.operands[0]]
    while todo:
        v = todo.pop()
        if isinstance(v, ir.BlockArgument):
            continue
        o = v.owner
        names.append(o.name)
        if not _is_pop(o):
            todo.extend(o.operands)
    return names


def _pop_stores(kernel):
    """Per pop: (whether its store is masked, the names of the ops feeding the store)."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_after_cross_lane(kernel))
        got = []
        for o in walk(m.operation):
            if _is_pop(o):
                store, names = _store_of(o)
                got.append((store is not None and _masked(store), names))
        return got


def every_permute_transfer_is_in_bounds():
    """The rounds' rows are checked inside the banks before any is emitted, so no transfer of a
    push or a pop carries a bounds check. Breaks if the permute drops `in_bounds`."""
    for kernel in (SPLIT, CUMSUM):
        with ir.Context() as ctx, ir.Location.unknown():
            ctx.allow_unregistered_dialects = True
            m = ir.Module.parse(_after_cross_lane(kernel))
            got = [_in_bounds(o) for ops, _ in _unit_blocks(m) for o in ops if o.name in TRANSFERS]
            assert got and all(all(b) for b in got), (os.path.basename(kernel), got)


def a_partial_round_stores_under_its_mask_and_reads_nothing():
    """ptr_div_only_axis_split's round writes only some lanes of each row: its pop's store is masked
    to them, with no read of the row and no select. Breaks if a partial pop read-modify-writes."""
    pops = _pop_stores(SPLIT)
    assert any(masked for masked, _ in pops), pops
    for masked, names in pops:
        assert masked, names
        assert TRANSFERS[0] not in names and "arith.select" not in names, names


def an_i64_partial_round_joins_its_high_word_under_the_mask():
    """An i64 pop's high word joins the low word already in the row, so it reads the row, and its
    store is masked to the round's lanes too. Breaks if the join's store writes every lane."""
    pops = _pop_stores(PERMUTE_I64)
    assert all("arith.select" not in names for _, names in pops), pops
    low = sum(1 for masked, names in pops if masked and TRANSFERS[0] not in names)
    high = sum(1 for masked, names in pops if masked and TRANSFERS[0] in names)
    assert low and high == low, (low, high)


def a_partial_round_moves_a_register_of_rows():
    """ptr_div_only_axis_split's partial round runs its rows on at step 1 and writes some lanes of a
    row, so a push and a pop carry as many as a register group of the i64 lane numbers that pick
    those lanes holds (ISA_MAX_LMUL * VLEN / 64). Breaks if it goes row by row, or stops at LMUL 1."""
    from pytorchsim_triton_compiler import machine
    per_reg = machine.ISA_MAX_LMUL * machine.VLEN_BITS // 64
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_after_cross_lane(SPLIT))
        widths = {(o.name, ir.VectorType(o.results[0].type if _is_pop(o) else o.operands[0].type)
                   .shape[0]) for o in walk(m.operation) if _is_push(o) or _is_pop(o)}
    assert widths == {("vcix.v.i", per_reg), ("vcix.iv", per_reg)}, widths


def a_pattern_is_set_once_before_the_pushes_it_shapes():
    """The split kernel's rounds share a pattern: each run sets it once with xlu_set_pattern, outside
    the loop of its pushes, and the pushes carry rows only, as the v6e's vset.pattern.permute before
    its vperms. Breaks if a push carries its own pattern again."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_after_cross_lane(SPLIT))
        ops = list(walk(m.operation))
        sets = [o for o in ops if _is_set(o)]
        pushes = [o for o in ops if _is_push(o)]
    assert sets and pushes, (len(sets), len(pushes))
    loops = ("scf.for", "affine.for")
    for o in pushes:
        loop = o.parent
        assert loop.name in loops and not any(x.parent == loop for x in sets), loop.name


CASES = [every_permute_transfer_is_in_bounds, a_partial_round_moves_a_register_of_rows,
         a_pattern_is_set_once_before_the_pushes_it_shapes,
         a_partial_round_stores_under_its_mask_and_reads_nothing,
         an_i64_partial_round_joins_its_high_word_under_the_mask]


def main():
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
