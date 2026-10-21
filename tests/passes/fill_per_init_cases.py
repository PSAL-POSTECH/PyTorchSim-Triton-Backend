"""torchsim-bufferize on a fill that several reduces take as their init: each init is filled where
it is read, with no copy of one filled buffer (attention's row maxes, one -inf after cse, kept
their seed in a copy vectorize could not take for a constant); a fill one op takes stays one.
    python3 tests/passes/fill_per_init_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.passmanager import PassManager  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p04_bufferize as p04  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

T, R = "tensor<4x8xf32>", "tensor<4xf32>"
ROWS = "affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>"


def _row_max(name, x, init):
    return (f"  {name} = linalg.generic {{indexing_maps = [{ROWS}], "
            f'iterator_types = ["parallel", "reduction"]}} ins({x} : {T}) outs({init} : {R}) {{\n'
            f"  ^bb0(%a: f32, %o: f32):\n    %s = arith.maxnumf %a, %o : f32\n"
            f"    linalg.yield %s : f32\n  }} -> {R}\n")


def _module(reduces):
    body = "".join(_row_max(f"%r{k}", f"%x{k % 2}", "%t") for k in range(reduces))
    outs = ", ".join(f"%r{k}" for k in range(reduces))
    types = ", ".join([R] * reduces)
    return (f"func.func @f(%x0: {T}, %x1: {T}) -> ({types}) {{\n"
            "  %ninf = arith.constant 0xFF800000 : f32\n"
            f"  %e = tensor.empty() : {R}\n"
            f"  %t = linalg.fill ins(%ninf : f32) outs(%e : {R}) -> {R}\n"
            + body + f"  return {outs} : {types}\n}}\n")


def _bufferized(reduces):
    """(the fills, the copies, and for each reduce whether the op right before it fills its
    init) after the pass's split of fills and one-shot-bufferize."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_module(reduces))
        for f in [o for o in walk(m.operation) if o.name == "func.func"]:
            p04._rewrite_fills_into_inits(f)
        PassManager.parse(p04.BUFFERIZE_PIPELINE).run(m.operation)
        assert m.operation.verify()
        ops = list(walk(m.operation))
        fills = [o for o in ops if o.name == "linalg.fill"]
        copies = [o for o in ops if o.name in ("memref.copy", "linalg.copy")
                  or (o.name == "linalg.generic" and "reduction" not in
                      str(o.attributes["iterator_types"]))]
        filled = []
        for o in ops:
            if o.name != "linalg.generic" or "reduction" not in str(o.attributes["iterator_types"]):
                continue
            block = [x.operation for x in o.block.operations]
            at = next(i for i, x in enumerate(block) if x == o)
            before = block[at - 1] if at else None
            filled.append(before is not None and before.name == "linalg.fill"
                          and before.operands[-1] == o.operands[-1])
        return len(fills), len(copies), filled


def a_fill_two_reduces_take_is_made_at_each():
    fills, copies, filled = _bufferized(2)
    assert (fills, copies, filled) == (2, 0, [True, True]), (fills, copies, filled)


def a_fill_one_reduce_takes_stays_one():
    fills, copies, filled = _bufferized(1)
    assert (fills, copies, filled) == (1, 0, [True]), (fills, copies, filled)


CASES = [a_fill_two_reduces_take_is_made_at_each, a_fill_one_reduce_takes_stays_one]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
