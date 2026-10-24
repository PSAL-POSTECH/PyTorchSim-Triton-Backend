"""pytorchsim-fuse makes a mask that several walks read in each walk's body, so no tile holds it and
pytorchsim-fit-to-numel cuts every scratchpad buffer to the 48 columns the mask lets through.
    python3 tests/passes/index_maker_cases.py   exit 0 iff all hold"""
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
from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import is_a_scratchpad_buffer  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

SHARED = os.path.join(ROOT, "tests", "cases", "mechanism", "mask", "mask_shared_by_two_reduces.py")


def _transform(d):
    subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), SHARED,
                    "--workdir", d, "--to", "transform"], check=False, capture_output=True)
    return lambda name: ir.Module.parse(open(os.path.join(d, "08-passes", name)).read())


def no_tile_holds_the_mask():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = _transform(d)("05-pytorchsim-fuse.mlir")
        makers = [o for o in walk(m.operation) if o.name == "linalg.generic"
                  and len(o.operands) == len(o.results)]
        assert not makers, f"{len(makers)} generic(s) still make a tile from indices alone"


def every_buffer_is_cut_to_the_mask():
    with tempfile.TemporaryDirectory() as d, ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = _transform(d)("08-pytorchsim-fit-to-numel.mlir")
        shapes = [list(ir.MemRefType(o.results[0].type).shape) for o in walk(m.operation)
                  if len(o.results) == 1 and is_a_scratchpad_buffer(o.results[0])]
        tiles = [s for s in shapes if len(s) == 2]
        assert tiles and all(s == [128, 48] for s in tiles), f"buffers {shapes}, want each tile 128x48"


CASES = [no_tile_holds_the_mask, every_buffer_is_cut_to_the_mask]


def main():
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
