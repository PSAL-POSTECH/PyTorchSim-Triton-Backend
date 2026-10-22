"""torchsim-vectorize (p05): a scalar memref.store into a one-element scratchpad buffer leaves as a
vector<1> store of it, which every lane takes into its own bank; a DRAM store is left alone.
    python3 tests/passes/uniform_scalar_write_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p05_vectorize  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import in_scratchpad  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

WRITES = """func.func @f(%x: f32, %h: bf16, %a: memref<f32, 1>, %b: memref<1xbf16, 1>, %d: memref<f32>) {
  %c0 = arith.constant 0 : index
  memref.store %x, %a[] : memref<f32, 1>
  memref.store %h, %b[%c0] : memref<1xbf16, 1>
  memref.store %x, %d[] : memref<f32>
  return
}
"""


def _writes(m):
    """(scratchpad?, the type of what each op writes) for every op writing a memref, off types."""
    from pytorchsim_triton_compiler.dialects.torchsim import effects_on
    got = []
    for op in walk(m.operation):
        for v in op.operands:
            if isinstance(v.type, ir.MemRefType) and "write" in effects_on(op, v):
                data = [str(x.type) for x in op.operands
                        if not isinstance(x.type, (ir.MemRefType, ir.IndexType))]
                got.append((in_scratchpad(v.type), data))
    return sorted(got)


def a_scalar_written_into_the_scratchpad_is_a_vector_of_one():
    """Breaks if p05 leaves the scratchpad's scalar stores to a later pass: before, they leave p05 as
    they came, scalar, and lane 0's bank alone takes them."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(WRITES)
        p05_vectorize.run(m, lanes=128)
        got = _writes(m)
    assert got == [(False, ["f32"]), (True, ["vector<1xbf16>"]), (True, ["vector<1xf32>"])], got


def a_scalar_written_into_a_banked_buffer_is_still_refused():
    """A store into a buffer of more than one element addresses it as flat; making it a vector<1>
    would hide that from the check after it, so it is left for the check to refuse."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse("""func.func @f(%x: f32, %a: memref<128xf32, 1>) {
  %c3 = arith.constant 3 : index
  memref.store %x, %a[%c3] : memref<128xf32, 1>
  return
}
""")
        try:
            p05_vectorize.run(m, lanes=128)
        except p05_vectorize.NotVectorisable as e:
            assert "as if it were flat" in str(e), str(e)
            return
    raise AssertionError("not refused")


CASES = [a_scalar_written_into_the_scratchpad_is_a_vector_of_one,
         a_scalar_written_into_a_banked_buffer_is_still_refused]


if __name__ == "__main__":
    bad = 0
    for c in CASES:
        try:
            c()
            print(f"ok   {c.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {c.__name__}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if bad else 0)

