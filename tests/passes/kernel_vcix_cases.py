"""A kernel's vcix is lowered by its own pass, which runs alone when the module holds no cross-lane
move: bank_view reads the cast beside a buffer from the IR, so a cast of another buffer is never its.
    python3 tests/passes/kernel_vcix_cases.py        exit 0 iff every case holds"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lower_kernel_vcix  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lib_bank_layout, lib_ir  # noqa: E402

_MOD = """
module {
  memref.global @dst : memref<128x8xf32, 1>
  memref.global @src : memref<128x8xf32, 1>
  func.func @f() {
    %0 = memref.get_global @dst : memref<128x8xf32, 1>
    %other = "pytorchsim.bank_view"(%0) {offset = 0 : i64, sizes = array<i64: 1, 8>,
        strides = array<i64: 8, 1>} : (memref<128x8xf32, 1>) -> memref<1x8xf32, strided<[8, 1]>, 1>
    %1 = memref.get_global @src : memref<128x8xf32, 1>
    "pytorchsim.vcix"(%1) <{form = "iv", imm = 1 : i64, layouts = [#ttg.blocked<{sizePerThread = [1, 1],
        threadsPerWarp = [1, 128], warpsPerCTA = [1, 1], order = [1, 0]}>], opcode = 3 : i64,
        operandSegmentSizes = array<i32: 1, 0>}> : (memref<128x8xf32, 1>) -> ()
    return
  }
}
"""

_TARGET = types.SimpleNamespace(vector_lanes=128, vlen_bits=256)


def a_cast_of_another_buffer_is_not_read():
    from mlir.ir import Module
    module = Module.parse(_MOD)
    ops = list(lib_ir.walk(module.operation))
    _dst, src = [o.results[0] for o in ops if o.name == "memref.get_global"]
    lower_kernel_vcix.run(module, target=_TARGET)
    assert module.operation.verify()
    views = [o for o in lib_ir.walk(module.operation) if o.name == lib_bank_layout.BANK_VIEW_OP]
    assert any(v.operands[0] == src for v in views), "the push read the stale cast of @dst"
    pushes = [o for o in lib_ir.walk(module.operation) if o.name == "vcix.iv"]
    assert pushes and not any(o.name == "pytorchsim.vcix" for o in lib_ir.walk(module.operation))


CASES = [a_cast_of_another_buffer_is_not_read]


def main():
    from mlir.ir import Context, Location
    failed = 0
    for case in CASES:
        try:
            with Context() as ctx, Location.unknown():
                ctx.allow_unregistered_dialects = True
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
