"""An elementwise loop that only reads a dot's accumulator runs at the last K subtile's pops, its
staging waits kept ahead of every store, and so does a walk folding C's rows by an op of any order
(attention's scale, causal mask and row max); a loop reading other elements, writing the dot's own
operand, or folding by an op whose answer depends on the order, stays a loop marked why.
    python3 tests/passes/epilogue_at_pop_cases.py   exit 0 iff all"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run  # noqa: E402
from ir_helpers import global_root  # noqa: E402
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects.torchsim import TransferOp, is_load  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lib_matmul  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_matmul  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

TILE = os.path.join(ROOT, "tests", "cases", "mechanism", "tile")
AT_POP = os.path.join(TILE, "tile_gemm_epilogue_at_pop.py")
STAGING = os.path.join(TILE, "tile_gemm_epilogue_shared_staging.py")
INTO_OPERAND = os.path.join(TILE, "tile_gemm_epilogue_into_operand.py")
# Causal flash attention: QK^T is followed by a walk over its rows -- the scale, the causal mask's
# select, the row's max carried along it and stored a row at a time.
ROW_FOLD = os.path.join(ROOT, "tests", "cases", "mechanism", "loops",
                        "loop_08_trip_from_program_id_tog.py")
# A dot's rows cast to i32 and folded by xor, seeded with the xor's identity.
ROW_XOR = os.path.join(TILE, "tile_gemm_row_xor_at_pop.py")


def _compile(kernel, d, to):
    """Compile to `to`; the IR dumps, not the kernel's own expect_ir verdict, are what is read."""
    subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                    "--workdir", d, "--to", to], check=False, capture_output=True)


def _parse(path):
    return ir.Module.parse(open(path).read())


def _names(module):
    return [o.name for o in walk(module.operation)]


def _holds_pop(block):
    return any(o.name == "vcix.v.i" for blk_op in block.operations
               for o in [blk_op.operation] + walk(blk_op.operation))


def the_epilogue_runs_beside_the_pops():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        _compile(AT_POP, d, "lower")
        m = _parse(os.path.join(d, "09-passes", "01-tritonpytorchsim-lower-matmul-tpuv6e.mlir"))
        assert "affine.vector_load" not in _names(m), "a loop still reads the accumulator"
        casts = [o for o in walk(m.operation) if o.name == "arith.truncf"]
        assert casts, "no cast left"
        for o in casts:
            blk = o.block
            while blk.owner.name == "scf.if":
                blk = blk.owner.operation.block
            assert _holds_pop(blk), f"a cast stands apart from the pops: {o}"


def a_loop_reading_another_element_stays_a_loop():
    """The cast's store is turned to write C's transposed element: its map's results reversed,
    whether the walk indexes C by two counters or by one run of a bank ([0, i])."""
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        _compile(AT_POP, d, "transform")
        m = _parse(os.path.join(d, "08-transformed.mlir"))
        stores = [o for o in walk(m.operation) if o.name == "affine.vector_store"
                  and isinstance(o.operands[0].type, ir.VectorType)
                  and ir.VectorType(o.operands[0].type).element_type == ir.BF16Type.get()]
        assert len(stores) == 1, len(stores)
        st = stores[0]
        at = ir.AffineMapAttr(st.attributes["map"]).value
        turned = ir.AffineMap.get(at.n_dims, at.n_symbols, list(at.results)[::-1])
        assert turned != at, at
        st.attributes["map"] = ir.AffineMapAttr.get(turned)
        lower_matmul.run(m, target=machine.target())
        names = _names(m)
        assert names.count("arith.truncf") == 1, names.count("arith.truncf")
        assert names.count("affine.vector_load") == 2, names.count("affine.vector_load")


def every_staging_store_follows_the_wait_on_its_last_mvout():
    """A regression guard, not a pin (the tree before the fusion passes it too): a store into a
    staging buffer follows the wait of an async MVOUT draining it; a synchronous one is done."""
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        _compile(STAGING, d, "lower")
        m = _parse(os.path.join(d, "09-passes", "01-tritonpytorchsim-lower-matmul-tpuv6e.mlir"))
        outs = [o for o in walk(m.operation) if o.name == TransferOp.OPERATION_NAME
                and not is_load(o.opview)]
        pending_after = [o for o in outs if "async" in o.attributes
                         and ir.BoolAttr(o.attributes["async"]).value]
        staging = {global_root(o.opview.sram) for o in outs}
        assert staging, "no MVOUT"
        pending, stores = set(), 0
        for _ in range(2):
            for o in walk(m.operation):
                if o.name == "torchsim.wait":
                    pending.discard(o.operands[0])
                elif o in pending_after:
                    pending.add(o.opview.sem)
                elif o.name in ("vector.transfer_write", "affine.vector_store") \
                        and global_root(o.operands[1]) in staging:
                    stores += 1
                    assert not pending, f"a staging store ahead of its wait: {o}"
        assert stores, "no staging store"


def a_loop_writing_the_dots_operand_stays_a_loop_and_says_why():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        _compile(INTO_OPERAND, d, "lower")
        m = _parse(os.path.join(d, "09-passes", "01-tritonpytorchsim-lower-matmul-tpuv6e.mlir"))
        kept = [o for o in walk(m.operation) if o.name == "affine.for"
                and lib_matmul.KEPT_APART in o.attributes]
        assert len(kept) == 1 and "torchsim.dot" in ir.StringAttr(
            kept[0].attributes[lib_matmul.KEPT_APART]).value, kept
        assert any(o.name == "arith.truncf" for o in walk(kept[0])), "the cast left its loop"
        log = open(os.path.join(d, "stage.log")).read().splitlines()
        assert any(lib_matmul.KEPT_APART in line for line in log), "stage.log does not say why"


def _row_walk(m):
    """(QK^T's C, the walk's loop over C's rows, the op its innermost loop folds each row by)."""
    dot = next(o for o in walk(m.operation) if o.name == "torchsim.dot")
    C = global_root(dot.operands[2])
    inner = [o for o in walk(m.operation) if o.name == "affine.for" and len(o.results)
             and any(x.name == "affine.vector_load" and global_root(x.operands[0]) == C for x in walk(o))]
    assert len(inner) == 1, inner
    outer = inner[0].parent
    fold = inner[0].regions[0].blocks[0].operations[-1].operands[0].owner
    return C, outer, fold


def _from_pops(m):
    """The names of the ops a pop's value reaches, through any op that makes a value of it."""
    seen, todo = set(), [o.results[0] for o in walk(m.operation) if o.name == "vcix.v.i"]
    names = set()
    while todo:
        v = todo.pop()
        for u in v.uses:
            o = u.owner
            if o in seen:
                continue
            seen.add(o)
            names.add(o.name)
            todo.extend(o.results)
    return names


def _reads(m, C):
    return [o for o in walk(m.operation) if o.name in ("affine.vector_load", "vector.load")
            and global_root(o.operands[0]) == C]


def a_walk_folding_the_rows_runs_at_the_pops_mask_and_max_with_it():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        _compile(ROW_FOLD, d, "transform")
        m = _parse(os.path.join(d, "08-transformed.mlir"))
        C, _, fold = _row_walk(m)
        assert fold.name == "arith.maxnumf", fold.name
        lower_matmul.run(m, target=machine.target())
        assert m.operation.verify()
        assert not _reads(m, C), "a loop still reads QK^T's C back"
        reached = _from_pops(m)
        assert {"arith.mulf", "arith.select", "arith.maxnumf"} <= reached, sorted(reached)
        assert not [o for o in walk(m.operation) if lib_matmul.KEPT_APART in o.attributes]


def a_walk_summing_the_rows_stays_a_loop_and_says_why():
    """The fold made an addf: the pops reach a row's elements in another order, and fewer at a
    time, than the walk, and a float sum's answer depends on the order."""
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        _compile(ROW_FOLD, d, "transform")
        m = _parse(os.path.join(d, "08-transformed.mlir"))
        C, outer, fold = _row_walk(m)
        with ir.InsertionPoint(fold):
            add = ir.Operation.create("arith.addf", results=[fold.results[0].type],
                                      operands=list(fold.operands))
        fold.results[0].replace_all_uses_with(add.results[0])
        fold.erase()
        lower_matmul.run(m, target=machine.target())
        assert m.operation.verify()
        assert lib_matmul.KEPT_APART in outer.attributes, "the walk is not marked"
        why = ir.StringAttr(outer.attributes[lib_matmul.KEPT_APART]).value
        assert "any order" in why, why
        assert len(_reads(m, C)) == 1, "the walk does not read C back"
        assert "arith.select" not in _from_pops(m), "the mask left the walk"


def _inner_fold(m, C):
    """The loop of the walk over C's rows that folds them, and the op it folds them by."""
    inner = [o for o in walk(m.operation) if o.name == "affine.for" and len(o.results)
             and any(x.name == "affine.vector_load" and global_root(x.operands[0]) == C for x in walk(o))]
    assert len(inner) == 1, inner
    return inner[0], inner[0].regions[0].blocks[0].operations[-1].operands[0].owner


def a_walk_xoring_the_rows_from_the_identity_runs_at_the_pops():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        _compile(ROW_XOR, d, "transform")
        m = _parse(os.path.join(d, "08-transformed.mlir"))
        dot = next(o for o in walk(m.operation) if o.name == "torchsim.dot")
        C = global_root(dot.operands[2])
        _, fold = _inner_fold(m, C)
        assert fold.name == "arith.xori", fold.name
        lower_matmul.run(m, target=machine.target())
        assert m.operation.verify()
        assert not _reads(m, C), "a loop still reads the dot's C back"
        assert "arith.xori" in _from_pops(m), "the xor left the pops"


def a_walk_xoring_the_rows_from_another_seed_stays_a_loop_and_says_why():
    """The seed made 1: a row folds its seed once a lane, and the pops are narrower than the walk,
    so an xor would fold it another number of times."""
    from mlir.dialects import arith
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        _compile(ROW_XOR, d, "transform")
        m = _parse(os.path.join(d, "08-transformed.mlir"))
        dot = next(o for o in walk(m.operation) if o.name == "torchsim.dot")
        C = global_root(dot.operands[2])
        inner, _ = _inner_fold(m, C)
        seed = inner.opview.inits[0]
        ty = ir.VectorType(seed.type)
        with ir.InsertionPoint(inner.parent):
            one = arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(
                ty, ir.IntegerAttr.get(ty.element_type, 1))).result
        k = [x for x in inner.operands].index(seed)
        inner.operands[k] = one
        one.owner.move_before(inner)
        # At LMUL 1 the pops are a register of the walk's group: narrower than the walk.
        lower_matmul.run(m, target=machine.target(), lmul=1)
        assert m.operation.verify()
        assert lib_matmul.KEPT_APART in inner.parent.attributes, "the walk is not marked"
        why = ir.StringAttr(inner.parent.attributes[lib_matmul.KEPT_APART]).value
        assert "identity" in why, why
        assert len(_reads(m, C)) == 1, "the walk does not read C back"


def _lowered_ops(transformed, as_allocs):
    """The ops lower_matmul leaves of `transformed`, or of its alloc twin, but a buffer's own."""
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime"))
    from alloc_twin import to_allocs
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(transformed)
        if as_allocs:
            to_allocs(m)
        lower_matmul.run(m, target=machine.target())
        life = ("memref.get_global", "memref.alloc", "memref.dealloc", "memref.global")
        return [o.name for o in walk(m.operation) if o.name not in life]


def allocated_buffers_lower_as_their_globals_do():
    """Breaks if lower_matmul takes the dealloc of an allocated C (#230) for a read after the dot:
    the sum's last write to C, left out on globals (nothing but the epilogue reads it), stays."""
    with tempfile.TemporaryDirectory() as d:
        _compile(AT_POP, d, "transform")
        text = open(os.path.join(d, "08-transformed.mlir")).read()
    want, got = _lowered_ops(text, False), _lowered_ops(text, True)
    assert got == want, (len(got), len(want), [x for x in set(got) if got.count(x) != want.count(x)])


CASES = [allocated_buffers_lower_as_their_globals_do,
         the_epilogue_runs_beside_the_pops,
         a_loop_reading_another_element_stays_a_loop,
         every_staging_store_follows_the_wait_on_its_last_mvout,
         a_loop_writing_the_dots_operand_stays_a_loop_and_says_why,
         a_walk_folding_the_rows_runs_at_the_pops_mask_and_max_with_it,
         a_walk_summing_the_rows_stays_a_loop_and_says_why,
         a_walk_xoring_the_rows_from_the_identity_runs_at_the_pops,
         a_walk_xoring_the_rows_from_another_seed_stays_a_loop_and_says_why]


def main():
    return run(CASES)


if __name__ == "__main__":
    sys.exit(main())
