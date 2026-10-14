"""A dot whose accumulator holds zero writes its first K subtile's pops and loses the zero fill and
copy that set it; an accumulator that is loaded, carried round a K loop, or filled non-zero keeps
both and adds every pop.   python3 tests/passes/zero_accumulator_cases.py   exit 0 iff all hold"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lower_matmul  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

CASES_DIR = os.path.join(ROOT, "tests", "cases")
STAGING = os.path.join(CASES_DIR, "mechanism", "tile", "tile_gemm_epilogue_shared_staging.py")
LOADED = os.path.join(CASES_DIR, "mechanism", "tile", "tile_gemm_streamed_accumulator.py")
K_LOOP = os.path.join(CASES_DIR, "smoke", "gemm.py")


def _transformed(kernel, d):
    """The kernel's 08-transformed.mlir, parsed; its expect_ir verdict is not what is read."""
    subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                    "--workdir", d, "--to", "transform"], check=False, capture_output=True)
    return ir.Module.parse(open(os.path.join(d, "08-transformed.mlir")).read())


def _root(v):
    while True:
        o = getattr(v, "owner", None)
        o = getattr(o, "operation", o)
        if not isinstance(o, ir.Operation):
            return v
        if o.name == "memref.get_global":
            return ir.FlatSymbolRefAttr(o.attributes["name"]).value
        if not len(o.operands):
            return v
        v = o.operands[0]


def _lower(m):
    """(accumulator roots, every loop store into one of them after the matmul lowering) for module `m`."""
    roots = {_root(o.operands[2]) for o in walk(m.operation) if o.name == "torchsim.dot"}
    assert roots, "no dot"
    lower_matmul.run(m, target=machine.target())
    assert m.operation.verify()
    inits = [o for o in walk(m.operation)
             if o.name == "affine.vector_store" and _root(o.operands[1]) in roots]
    return roots, inits


def _pops(m):
    """(pops written to C as popped, pops added into C) -- by the users of each pop's value."""
    written = added = 0
    for o in walk(m.operation):
        if o.name != "vcix.v.i" or ir.IntegerAttr(o.attributes["opcode"]).value != 2:
            continue
        users = [u.owner for u in o.results[0].uses]
        assert len(users) == 1 and users[0].name == "arith.bitcast", users
        names = [u.owner.name for u in users[0].results[0].uses]
        written += names.count("vector.transfer_write")
        added += names.count("arith.addf")
    return written, added


def a_zero_accumulator_loses_its_fill_and_copy():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = _transformed(STAGING, d)
        _, inits = _lower(m)
        assert not inits, f"{len(inits)} loop store(s) still fill or copy into an accumulator"


def the_first_k_pops_do_not_read_c():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = _transformed(STAGING, d)
        _lower(m)
        written, added = _pops(m)
        assert written and added, (written, added)


def _keeps_read_add_write(m, inits_left):
    roots, inits = _lower(m)
    written, added = _pops(m)
    assert written == 0 and added, (written, added)
    assert bool(inits) == inits_left, inits


def a_loaded_accumulator_adds_every_pop():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        _keeps_read_add_write(_transformed(LOADED, d), False)


def a_k_loop_accumulator_adds_every_pop_and_keeps_its_fill():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        _keeps_read_add_write(_transformed(K_LOOP, d), True)


def a_non_zero_fill_is_kept_and_added_to():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = _transformed(STAGING, d)
        fills = [o for o in walk(m.operation) if o.name == "affine.vector_store"
                 and o.operands[0].owner.name == "arith.constant"]
        assert len(fills) == 1, fills
        cst = fills[0].operands[0].owner
        ty = cst.results[0].type
        one = ir.DenseElementsAttr.get_splat(
            ty, ir.FloatAttr.get(ir.VectorType(ty).element_type, 1.0))
        with ir.InsertionPoint(cst):
            new = ir.Operation.create("arith.constant", results=[ty], attributes={"value": one})
        fills[0].operands[0] = new.results[0]
        _keeps_read_add_write(m, True)


CASES = [a_zero_accumulator_loses_its_fill_and_copy,
         the_first_k_pops_do_not_read_c,
         a_loaded_accumulator_adds_every_pop,
         a_k_loop_accumulator_adds_every_pop_and_keeps_its_fill,
         a_non_zero_fill_is_kept_and_added_to]


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
