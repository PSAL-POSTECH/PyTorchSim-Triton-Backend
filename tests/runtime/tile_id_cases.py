"""The cycle binary's walk against the TOG's: every replayed loop makes exactly one trip from its own
start, and a segment is the tile the skeleton stamped on it, refused when the two walks disagree.
    python3 tests/runtime/tile_id_cases.py        exit 0 iff every case gets what it names"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytorchsim_triton_compiler  # noqa: E402,F401
from pytorchsim_triton_compiler.contract.gem5_samples import SAMPLE_OPEN  # noqa: E402
from pytorchsim_triton_compiler.dialect import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.passes.utils.ir_utils import trip_count  # noqa: E402
from pytorchsim_triton_compiler.togsim import build_tog as bt  # noqa: E402
from mlir.dialects import affine, arith, llvm, scf  # noqa: E402


def _kernel(body):
    """A @kernel(dram, sram, x) whose block `body(args, const)` fills; the module and the context."""
    from mlir.dialects import func
    from mlir.ir import (Context, F32Type, IndexType, InsertionPoint, IntegerAttr, Location,
                         MemRefType, Module)
    from mlir.dialects import arith
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        m = Module.create()
        f32, idx = F32Type.get(), IndexType.get()
        dram = MemRefType.get([64], f32)
        sram = MemRefType.get([8, 8], f32, memory_space=IntegerAttr.get(idx, 1))
        with InsertionPoint(m.body):
            fn = func.FuncOp("kernel", ([dram, sram, f32], []))
        blk = fn.add_entry_block()
        with InsertionPoint(blk):
            body(list(blk.arguments),
                 lambda v, ty=idx: arith.ConstantOp(ty, IntegerAttr.get(ty, v)).result)
            func.ReturnOp([])
    return m, ctx


def _dma(args, const):
    pytorchsim.TransferOp(None, args[0], const(0), args[1], const(0), const(0), [], [],
                          pytorchsim.direction_attr(pytorchsim.Direction.load), 0, [8, 1], async_=True)


def _compute(args):
    from mlir.dialects import arith
    return arith.AddFOp(args[2], args[2]).result


def _loop_trips(kind, lb, ub, step, width=None):
    """Trips of a replayed loop (a DMA and a compute) after the cycle binary's collapse; `kind` is
    affine.AffineForOp or scf.ForOp."""
    from mlir.ir import InsertionPoint, IntegerType

    def body(args, const):
        if kind is affine.AffineForOp:
            loop = affine.AffineForOp(lb, ub, step)
            yield_ = affine.AffineYieldOp
        else:
            ty = IntegerType.get_signless(width) if width else None
            c = (lambda v: const(v, ty)) if ty else const
            loop = scf.ForOp(c(lb), c(ub), c(step))
            yield_ = scf.YieldOp
        with InsertionPoint(loop.body):
            _dma(args, const)
            _compute(args)
            yield_([])

    m, ctx = _kernel(body)
    with ctx:
        bt.stamp_tile_ids(_built(m))
        bt.build_tog_and_mutate(m)
        loop = [o for o in bt._all_ops(bt._find_kernel(m)) if isinstance(o.operation.opview, kind)][0]
        return trip_count(loop)


def _built(m):
    builder = bt.TogBuilder()
    bt._build(m, builder)
    return builder


def _two_segments(tamper=None):
    """Two segments a DMA parts, stamped as the skeleton stamps them, then `tamper(ops)`; the
    tile ids the cycle binary's markers carry, or its refusal."""
    from mlir.ir import IntegerAttr, IntegerType

    def body(args, const):
        _compute(args)
        _dma(args, const)
        _compute(args)

    m, ctx = _kernel(body)
    with ctx:
        if tamper != "unstamped":
            bt.stamp_tile_ids(_built(m))
        ops = list(bt._all_ops(bt._find_kernel(m)))
        adds = [o for o in ops if isinstance(o.operation.opview, arith.AddFOp)]
        i64 = IntegerType.get_signless(64)
        if tamper == "dropped":
            del adds[1].operation.attributes[bt.TILE_ID]
        elif tamper == "swapped":
            adds[0].operation.attributes[bt.TILE_ID] = IntegerAttr.get(i64, 1)
            adds[1].operation.attributes[bt.TILE_ID] = IntegerAttr.get(i64, 0)
        elif tamper == "stray":
            stray = [o for o in ops if isinstance(o.operation.opview, arith.ConstantOp)][0]
            stray.operation.attributes[bt.TILE_ID] = IntegerAttr.get(i64, 0)
        try:
            bt.build_tog_and_mutate(m)
        except RuntimeError as e:
            return str(e)
        return [IntegerAttr(o.operation.attributes[bt.TILE_ID]).value
                for o in bt._all_ops(bt._find_kernel(m)) if isinstance(o.operation.opview, llvm.InlineAsmOp)
                and bt.TILE_ID in o.operation.attributes]


def _start_markers():
    """Each start marker's (tile id attribute, asm, constraints): it loads a0 with its tile id and
    a1 with the wrapper's call index before gem5's resetstats reads them."""
    from mlir.ir import IntegerAttr, StringAttr

    def body(args, const):
        _compute(args)
        _dma(args, const)
        _compute(args)

    m, ctx = _kernel(body)
    with ctx:
        bt.stamp_tile_ids(_built(m))
        bt.build_tog_and_mutate(m)
        return [(IntegerAttr(o.operation.attributes[bt.TILE_ID]).value,
                 StringAttr(o.operation.attributes["asm_string"]).value,
                 StringAttr(o.operation.attributes["constraints"]).value)
                for o in bt._all_ops(bt._find_kernel(m)) if isinstance(o.operation.opview, llvm.InlineAsmOp)
                and bt.TILE_ID in o.operation.attributes]


def _start(tile):
    return (tile, f"li a0, {tile}\nla a1, __pytorchsim_cycle_call\nld a1, 0(a1)\n" + SAMPLE_OPEN,
            "~{a0},~{a1},~{memory},~{sf.vcix_state}")


def _carried_empty_affine_for():
    """Whether the skeleton would erase an affine.for that only yields its carried value, which a
    later op reads: erasing it would leave that read with no definition."""
    from mlir.dialects import affine, arith
    from mlir.ir import InsertionPoint

    def body(args, const):
        loop = affine.AffineForOp(0, 4, 1, iter_args=[args[2]])
        with InsertionPoint(loop.body):
            affine.AffineYieldOp([loop.inner_iter_args[0]])
        arith.AddFOp(loop.results[0], args[2])

    m, ctx = _kernel(body)
    with ctx:
        loop = [o for o in bt._all_ops(bt._find_kernel(m))
                if isinstance(o.operation.opview, affine.AffineForOp)][0]
        return bt._is_empty_affine_for(loop)


def _empty_segment():
    m, ctx = _kernel(lambda args, const: None)
    with ctx:
        builder = bt.TogBuilder()
        builder.compute_nodes.append(bt.TOGComputeNode(bt.SEGMENT_COMPUTE))
        try:
            bt.stamp_tile_ids(builder)
        except RuntimeError as e:
            return str(e)
    return "no refusal"


CASES = {
    "scf.for lb = step": (lambda: _loop_trips(scf.ForOp, 64, 1024, 64), 1),
    "scf.for lb > step": (lambda: _loop_trips(scf.ForOp, 96, 1024, 32), 1),
    "scf.for i32 lb = step": (lambda: _loop_trips(scf.ForOp, 64, 1024, 64, width=32), 1),
    "scf.for from 0": (lambda: _loop_trips(scf.ForOp, 0, 1024, 64), 1),
    "affine.for lb = step": (lambda: _loop_trips(affine.AffineForOp, 64, 1024, 64), 1),
    "affine.for short": (lambda: _loop_trips(affine.AffineForOp, 64, 96, 64), 1),
    "scf.for no trip refused": (lambda: _loop_trips(scf.ForOp, 64, 64, 64), "makes no trip"),
    "scf.for ub < lb refused": (lambda: _loop_trips(scf.ForOp, 128, 64, 64), "makes no trip"),
    "affine.for no trip refused": (lambda: _loop_trips(affine.AffineForOp, 64, 64, 64), "makes no trip"),
    "carried affine.for kept": (_carried_empty_affine_for, False),
    "markers carry ids": (lambda: _two_segments(), [0, 1]),
    "unstamped refused": (lambda: _two_segments("unstamped"), "not one stamped id"),
    "dropped id refused": (lambda: _two_segments("dropped"), "not one stamped id"),
    "swapped ids refused": (lambda: _two_segments("swapped"), "not 0..1"),
    "stray id refused": (lambda: _two_segments("stray"), "outside every segment"),
    "empty segment refused": (_empty_segment, "holds no op"),
    "markers hand gem5 tile and call": (_start_markers, [_start(0), _start(1)]),
}


def main():
    failed = 0
    for name, (case, want) in CASES.items():
        try:
            got = case()
        except Exception as e:  # noqa: BLE001
            got = f"{type(e).__name__}: {e}"
        ok = (want in got) if isinstance(want, str) and isinstance(got, str) else got == want
        failed += not ok
        print(f"{name:22} {'PASS' if ok else 'FAIL'}  got {str(got)[:110]}  want {want}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
