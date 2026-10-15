"""tritonpytorchsim-reduce-in-loop on the carried-combine-then-reduce loop: what moves the reduce into
the loop, and what leaves it -- another combiner, a seed that does not fold back, another reader.
    python3 tests/passes/reduce_in_loop_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    p02_materialize_layouts, p02a_reduce_in_loop as p02a, p03_fuse, transform)
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

ID = "affine_map<(d0, d1) -> (d0, d1)>"
T, ROW = "tensor<8x16xf32>", "tensor<8xf32>"


def _loop(comb="arith.addf", red="arith.addf", seed="0.0", red_seed=None, extra="", tail=""):
    """`acc = comb(acc, t)` over four chunks into an [8, 16] tile, then `red` over axis 1."""
    red_seed = seed if red_seed is None else red_seed
    return (f"func.func @f(%t: {T}) -> {ROW} {{\n  %c0 = arith.constant 0 : index\n"
            "  %c1 = arith.constant 1 : index\n  %c4 = arith.constant 4 : index\n"
            f"  %s = arith.constant {seed} : f32\n  %s2 = arith.constant {red_seed} : f32\n"
            f"  %e = tensor.empty() : {T}\n  %a0 = linalg.fill ins(%s : f32) outs(%e : {T}) -> {T}\n"
            f"  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %a0) -> ({T}) {{\n"
            f"    %n = linalg.generic {{indexing_maps = [{ID}, {ID}, {ID}], iterator_types = "
            f'["parallel", "parallel"]}} ins(%acc, %t : {T}, {T}) outs(%e : {T}) {{\n'
            f"    ^bb0(%x: f32, %y: f32, %o: f32):\n      %v = {comb} %x, %y : f32\n"
            f"      linalg.yield %v : f32\n    }} -> {T}\n" + extra +
            f"    scf.yield %n : {T}\n  }}\n"
            f"  %re = tensor.empty() : {ROW}\n  %i0 = linalg.fill ins(%s2 : f32) outs(%re : {ROW}) -> {ROW}\n"
            f"  %y = linalg.reduce ins(%r : {T}) outs(%i0 : {ROW}) dimensions = [1]\n"
            f"    (%in: f32, %init: f32) {{\n      %w = {red} %in, %init : f32\n"
            f"      linalg.yield %w : f32\n    }}\n" + tail + f"  return %y : {ROW}\n}}\n")


def _run(text):
    m = ir.Module.parse(text)
    p02a.run(m)
    assert m.operation.verify(), m
    return m


def _moved(m):
    """Whether the loop now carries a row and the one reduce sits inside it."""
    loop = next(o for o in walk(m.operation) if o.name == "scf.for")
    reds = [o for o in walk(m.operation) if o.name == "linalg.reduce"]
    inside = [o for o in walk(loop) if o.name == "linalg.reduce"]
    return loop.results[0].type == ir.Type.parse(ROW) and len(reds) == 1 and len(inside) == 1


def a_sum_after_a_summing_loop_moves_in_and_no_tile_is_filled():
    m = _run(_loop())
    assert _moved(m), m
    fills = [o for o in walk(m.operation) if o.name == "linalg.fill"]
    assert all(f.results[0].type == ir.Type.parse(ROW) for f in fills), m


def a_max_after_a_max_loop_moves_in_from_any_seed():
    """max(c, c) folds to c whatever c is, so the seed need not be the neutral element."""
    assert _moved(_run(_loop("arith.maximumf", "arith.maximumf", seed="3.0")))


def a_seed_that_does_not_fold_back_stays():
    """1 + 1 is not 1: moving the sum in would drop the seed counted once per element."""
    assert not _moved(_run(_loop(seed="1.0")))


def unequal_seeds_stay():
    assert not _moved(_run(_loop("arith.maximumf", "arith.maximumf", seed="3.0", red_seed="4.0")))


def another_combiner_stays():
    assert not _moved(_run(_loop("arith.addf", "arith.maximumf")))


def an_accumulator_read_elsewhere_in_the_loop_stays():
    extra = f"    %d = tensor.extract %acc[%c0, %c0] : {T}\n"
    assert not _moved(_run(_loop(extra=extra)))


def a_loop_result_read_elsewhere_stays():
    tail = f"  %d = tensor.extract %r[%c0, %c0] : {T}\n"
    assert not _moved(_run(_loop(tail=tail)))


def the_pass_runs_after_layouts_and_before_fuse():
    """Fuse restates a reduce as a generic and folds the identity the rewrite leaves."""
    ps = [p for _, phase in transform.PHASES for p in phase]
    assert ps.index(p02_materialize_layouts) < ps.index(p02a) < ps.index(p03_fuse), ps


CASES = [a_sum_after_a_summing_loop_moves_in_and_no_tile_is_filled,
         a_max_after_a_max_loop_moves_in_from_any_seed,
         a_seed_that_does_not_fold_back_stays,
         unequal_seeds_stay,
         another_combiner_stays,
         an_accumulator_read_elsewhere_in_the_loop_stays,
         a_loop_result_read_elsewhere_stays,
         the_pass_runs_after_layouts_and_before_fuse]


def main():
    failed = 0
    for case in CASES:
        try:
            with ir.Context(), ir.Location.unknown():
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
