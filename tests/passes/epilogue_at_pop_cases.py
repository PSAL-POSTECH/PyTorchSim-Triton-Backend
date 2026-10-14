"""An elementwise loop that only reads a dot's accumulator runs at the last K subtile's pops, its
staging waits kept ahead of every store; a loop reading other elements, or writing the dot's own
operand, stays a loop marked why.   python3 tests/passes/epilogue_at_pop_cases.py   exit 0 iff all"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
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
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        _compile(AT_POP, d, "transform")
        m = _parse(os.path.join(d, "08-transformed.mlir"))
        stores = [o for o in walk(m.operation) if o.name == "affine.vector_store"
                  and isinstance(o.operands[0].type, ir.VectorType)
                  and ir.VectorType(o.operands[0].type).element_type == ir.BF16Type.get()]
        assert len(stores) == 1, len(stores)
        st = stores[0]
        i, j = st.operands[2], st.operands[3]
        st.operands[2], st.operands[3] = j, i
        lower_matmul.run(m, target=machine.target())
        names = _names(m)
        assert names.count("arith.truncf") == 1, names.count("arith.truncf")
        assert names.count("affine.vector_load") == 2, names.count("affine.vector_load")


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
        staging = {_root(o.opview.sram) for o in outs}
        assert staging, "no MVOUT"
        pending, stores = set(), 0
        for _ in range(2):
            for o in walk(m.operation):
                if o.name == "torchsim.wait":
                    pending.discard(o.operands[0])
                elif o in pending_after:
                    pending.add(o.opview.sem)
                elif o.name in ("vector.transfer_write", "affine.vector_store") \
                        and _root(o.operands[1]) in staging:
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


CASES = [the_epilogue_runs_beside_the_pops,
         a_loop_reading_another_element_stays_a_loop,
         every_staging_store_follows_the_wait_on_its_last_mvout,
         a_loop_writing_the_dots_operand_stays_a_loop_and_says_why]


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
