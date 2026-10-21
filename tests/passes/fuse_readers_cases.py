"""torchsim-fuse's reader-into-producer joins on IR no kernel reaches: the order readers join in
is the order of the tile's uses, whatever addresses the ops landed at.
    python3 tests/passes/fuse_readers_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p03_fuse as p03  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import op_of, walk  # noqa: E402

ID = "affine_map<(d0, d1) -> (d0, d1)>"
T = "tensor<4x4xf32>"


def _generic(name, x, init, body):
    return (f"  {name} = linalg.generic {{indexing_maps = [{ID}, {ID}], "
            f'iterator_types = ["parallel", "parallel"]}} ins({x} : {T}) outs({init} : {T}) {{\n'
            f"  ^bb0(%a: f32, %o: f32):\n    {body}\n    linalg.yield %s : f32\n  }} -> {T}\n")


TWO_READERS = (f"func.func @f(%x: {T}, %y: {T}, %w: {T}) -> ({T}, {T}) {{\n"
               f"  %e = tensor.empty() : {T}\n"
               + _generic("%p", "%x", "%e", "%s = arith.negf %a : f32")
               + _generic("%r1", "%p", "%y", "%s = arith.addf %a, %a : f32")
               + _generic("%r2", "%p", "%w", "%s = arith.mulf %a, %a : f32")
               + f"  return %r1, %r2 : {T}, {T}\n}}\n")


def _case(fn):
    def wrapped():
        with ir.Context() as ctx, ir.Location.unknown():
            ctx.allow_unregistered_dialects = True
            fn()
    wrapped.__name__ = fn.__name__
    return wrapped


@_case
def two_elementwise_readers_join_in_the_order_of_the_tiles_uses():
    kept, seen = [], set()
    for k in range(32):
        m = ir.Module.parse(TWO_READERS)
        kept.append(m)
        kept.extend(ir.Module.parse(TWO_READERS) for _ in range(k % 5))
        func = [o for o in walk(m.operation) if o.name == "func.func"][0]
        y, w = func.regions[0].blocks[0].arguments[1:3]
        p = [o for o in walk(m.operation) if o.name == "linalg.generic"][0].results[0]
        first = op_of(next(iter(p.uses)).owner)
        want = [y, w] if first.operands[1] == y else [w, y]
        p03._rewrite_readers_into_producers(func)
        assert m.operation.verify()
        (g,) = [o for o in walk(m.operation) if o.name == "linalg.generic"]
        outs = list(g.operands)[len(g.operands) - len(g.results):]
        assert outs == want, k
        seen.add(tuple(o == y for o in outs))
    assert len(seen) == 1, seen


CASES = [two_elementwise_readers_join_in_the_order_of_the_tiles_uses]


def main():
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {e!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
