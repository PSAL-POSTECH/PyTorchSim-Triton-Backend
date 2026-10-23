"""torchsim-bufferize on a generator (an empty tensor, a fill) that several ops write in place: each
writer gets its own and one-shot-bufferize makes no copy of one filled buffer -- reduces' inits (one
-inf after cse seeds attention's row maxes), a loop's iter_args, a fill one op writes and others
only read, and a fill before a loop that one op in it writes each trip; a generator only read, or
only one op outside a loop takes, stays one.
    python3 tests/passes/generator_per_write_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run  # noqa: E402
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.passmanager import PassManager  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p04_bufferize as p04  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

T, R = "tensor<4x8xf32>", "tensor<4xf32>"
ROWS = "affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>"
HEAD = (f"func.func @f(%x0: {T}, %x1: {T}) -> ({{types}}) {{{{\n"
        "  %ninf = arith.constant 0xFF800000 : f32\n"
        f"  %e = tensor.empty() : {R}\n"
        f"  %t = linalg.fill ins(%ninf : f32) outs(%e : {R}) -> {R}\n")


def _row_max(name, x, init):
    return (f"  {name} = linalg.generic {{indexing_maps = [{ROWS}], "
            f'iterator_types = ["parallel", "reduction"]}} ins({x} : {T}) outs({init} : {R}) {{\n'
            f"  ^bb0(%a: f32, %o: f32):\n    %s = arith.maxnumf %a, %o : f32\n"
            f"    linalg.yield %s : f32\n  }} -> {R}\n")


def _row_add(name, a, b):
    """An elementwise op that reads `a` and writes a new tensor: `a` is only read."""
    m = "affine_map<(d0) -> (d0)>"
    return (f"  %e_{name[1:]} = tensor.empty() : {R}\n"
            f"  {name} = linalg.generic {{indexing_maps = [{m}, {m}, {m}], "
            f'iterator_types = ["parallel"]}} ins({a}, {b} : {R}, {R}) outs(%e_{name[1:]} : {R}) {{\n'
            f"  ^bb0(%p: f32, %q: f32, %o: f32):\n    %s = arith.addf %p, %q : f32\n"
            f"    linalg.yield %s : f32\n  }} -> {R}\n")


def _loop(name, x):
    """A loop whose iter_args starts at %t and that writes it each trip."""
    return (f"  %c0_{name[1:]} = arith.constant 0 : index\n  %c1_{name[1:]} = arith.constant 1 : index\n"
            f"  %c2_{name[1:]} = arith.constant 2 : index\n"
            f"  {name} = scf.for %i_{name[1:]} = %c0_{name[1:]} to %c2_{name[1:]} step %c1_{name[1:]} "
            f"iter_args(%acc_{name[1:]} = %t) -> ({R}) {{\n"
            + _row_max(f"%m_{name[1:]}", x, f"%acc_{name[1:]}")
            + f"    scf.yield %m_{name[1:]} : {R}\n  }}\n")


def _written_each_trip(name, x):
    """A loop that writes %t, filled before it, in each trip, and adds what it makes."""
    k = name[1:]
    return (f"  %c0_{k} = arith.constant 0 : index\n  %c1_{k} = arith.constant 1 : index\n"
            f"  %c2_{k} = arith.constant 2 : index\n  %z_{k} = tensor.empty() : {R}\n"
            f"  {name} = scf.for %i_{k} = %c0_{k} to %c2_{k} step %c1_{k} "
            f"iter_args(%acc_{k} = %z_{k}) -> ({R}) {{\n"
            + _row_max(f"%m_{k}", x, "%t") + _row_add(f"%n_{k}", f"%acc_{k}", f"%m_{k}")
            + f"    scf.yield %n_{k} : {R}\n  }}\n")


def _module(body, outs):
    types = ", ".join([R] * len(outs))
    return HEAD.format(types=types) + body + f"  return {', '.join(outs)} : {types}\n}}\n"


def _bufferized(text):
    """(the fills, the copies) after the pass's split of generators and one-shot-bufferize."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(text)
        for f in [o for o in walk(m.operation) if o.name == "func.func"]:
            p04._rewrite_generators_per_write(f)
        PassManager.parse(p04.BUFFERIZE_PIPELINE).run(m.operation)
        assert m.operation.verify()
        ops = list(walk(m.operation))
        fills = [o for o in ops if o.name == "linalg.fill"]
        copies = [o for o in ops if o.name in ("memref.copy", "linalg.copy")]
        return len(fills), len(copies)


def _reduces(n):
    return _module("".join(_row_max(f"%r{k}", f"%x{k % 2}", "%t") for k in range(n)),
                   [f"%r{k}" for k in range(n)])


def a_fill_two_reduces_write_is_made_at_each():
    assert _bufferized(_reduces(2)) == (2, 0), _bufferized(_reduces(2))


def a_fill_three_reduces_write_is_made_at_each():
    assert _bufferized(_reduces(3)) == (3, 0), _bufferized(_reduces(3))


def a_fill_one_reduce_takes_stays_one():
    assert _bufferized(_reduces(1)) == (1, 0), _bufferized(_reduces(1))


def a_fill_two_loops_carry_is_made_at_each():
    got = _bufferized(_module(_loop("%l0", "%x0") + _loop("%l1", "%x1"), ["%l0", "%l1"]))
    assert got == (2, 0), got


def a_fill_one_op_writes_and_one_reads_is_made_for_the_writer():
    got = _bufferized(_module(_row_max("%r0", "%x0", "%t") + _row_add("%s0", "%t", "%r0"),
                              ["%r0", "%s0"]))
    assert got == (2, 0), got


def a_fill_only_read_stays_one():
    got = _bufferized(_module(_row_add("%s0", "%t", "%t") + _row_add("%s1", "%t", "%s0"),
                              ["%s0", "%s1"]))
    assert got == (1, 0), got


def a_fill_a_loop_writes_each_trip_is_made_in_the_loop():
    got = _bufferized(_module(_written_each_trip("%l0", "%x0"), ["%l0"]))
    assert got == (1, 0), got


def without_the_split_one_shot_copies():
    """The premise: one-shot-bufferize alone keeps one buffer and copies it for the other writer."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(_reduces(2))
        PassManager.parse(p04.BUFFERIZE_PIPELINE).run(m.operation)
        copies = [o for o in walk(m.operation) if o.name == "memref.copy"]
        assert len(copies) == 1, len(copies)


CASES = [a_fill_two_reduces_write_is_made_at_each, a_fill_three_reduces_write_is_made_at_each,
         a_fill_one_reduce_takes_stays_one, a_fill_two_loops_carry_is_made_at_each,
         a_fill_one_op_writes_and_one_reads_is_made_for_the_writer, a_fill_only_read_stays_one,
         a_fill_a_loop_writes_each_trip_is_made_in_the_loop, without_the_split_one_shot_copies]


def main():
    return run(CASES)


if __name__ == "__main__":
    sys.exit(main())
