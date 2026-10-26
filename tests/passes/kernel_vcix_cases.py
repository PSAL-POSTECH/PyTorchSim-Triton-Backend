"""convert-pytorchsim-extern-to-vcix writes a kernel's vcix on tiles as VCIX, once per register of each lane's
rows: bank_view reads the cast beside a buffer from the IR, so a cast of another buffer is never its; a form
the VCIX dialect has no op for is refused.
    python3 tests/passes/kernel_vcix_cases.py        exit 0 iff every case holds"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.dialects import pytorchsim_tpu  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import c16_convert_pytorchsim_extern_to_vcix  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lib_ir  # noqa: E402

_MOD = """
module {
  memref.global @dst : memref<128x8xf32, 1>
  memref.global @src : memref<128x8xf32, 1>
  func.func @f() {
    %0 = memref.get_global @dst : memref<128x8xf32, 1>
    %other = "pytorchsim_tpu.vpu.bank_view"(%0) {offset = 0 : i64, sizes = array<i64: 1, 8>,
        strides = array<i64: 8, 1>} : (memref<128x8xf32, 1>) -> memref<1x8xf32, strided<[8, 1]>, 1>
    %1 = memref.get_global @src : memref<128x8xf32, 1>
    "pytorchsim_extern.vcix"(%1) <{form = "iv", imm = 1 : i64,
        layouts = [#ttg.blocked<{sizePerThread = [1, 1],
        threadsPerWarp = [1, 128], warpsPerCTA = [1, 1], order = [1, 0]}>], opcode = 3 : i64,
        operandSegmentSizes = array<i32: 1, 0>}> : (memref<128x8xf32, 1>) -> ()
    return
  }
}
"""

_TARGET = types.SimpleNamespace(name="tpuv6e", vector_lanes=128, vlen_bits=256,
                                isa=["m", "a", "f", "d", "c", "v", "zvl256b", "pytorchsim_tpuv6e"],
                                kind="tpu")


def a_cast_of_another_buffer_is_not_read():
    from mlir.ir import Module
    module = Module.parse(_MOD)
    ops = list(lib_ir.walk(module.operation))
    _dst, src = [o.results[0] for o in ops if o.name == "memref.get_global"]
    c16_convert_pytorchsim_extern_to_vcix.run(module, target=_TARGET)
    assert module.operation.verify()
    views = [o for o in lib_ir.walk(module.operation) if o.name == pytorchsim_tpu.BANK_VIEW]
    assert any(v.operands[0] == src for v in views), "the push read the stale cast of @dst"
    pushes = [o for o in lib_ir.walk(module.operation) if o.name == "vcix.iv"]
    assert pushes and not any(o.name.startswith("pytorchsim_extern.vcix")
                              for o in lib_ir.walk(module.operation))


def a_v_i_writes_each_register_of_its_rows():
    """A v.i of 32 f32 rows a lane at VLEN 256: four vcix.v.i of eight elements, each written to its rows.
    Breaks if the rows are walked at another width (a unit's 32-bit crossing) or not at all."""
    from mlir.dialects import affine
    from mlir.ir import Module
    module = Module.parse("""
module {
  memref.global @dst : memref<128x32xf32, 1>
  func.func @f() {
    %0 = memref.get_global @dst : memref<128x32xf32, 1>
    "pytorchsim_extern.vcix"(%0) <{form = "v.i", imm = 0 : i64,
        layouts = [], opcode = 5 : i64, operandSegmentSizes = array<i32: 0, 1>}>
        : (memref<128x32xf32, 1>) -> ()
    return
  }
}""")
    c16_convert_pytorchsim_extern_to_vcix.run(module, target=_TARGET)
    assert module.operation.verify()
    ops = list(lib_ir.walk(module.operation))
    pops = [o for o in ops if o.name == "vcix.v.i"]
    loops = [o for o in ops if isinstance(o.opview, affine.AffineForOp)]
    assert len(pops) == 1 and len(loops) == 1, [o.name for o in ops]
    assert str(pops[0].results[0].type) == "vector<8xf32>", pops[0].results[0].type


def a_kernel_vcix_of_no_vcix_form_is_refused():
    """An iv with a dst names no op of the VCIX dialect: refused, not written."""
    from mlir.ir import Module
    module = Module.parse("""
module {
  memref.global @a : memref<128x8xf32, 1>
  func.func @f() {
    %0 = memref.get_global @a : memref<128x8xf32, 1>
    "pytorchsim_extern.vcix"(%0, %0) <{form = "iv", imm = 0 : i64, opcode = 1 : i64,
        layouts = [#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 128], warpsPerCTA = [1, 1],
        order = [1, 0]}>],
        operandSegmentSizes = array<i32: 1, 1>}> : (memref<128x8xf32, 1>, memref<128x8xf32, 1>) -> ()
    return
  }
}""")
    try:
        c16_convert_pytorchsim_extern_to_vcix.run(module, target=_TARGET)
    except NotImplementedError as e:
        assert "no op of that form" in str(e), e
        return
    raise AssertionError("an iv with a dst was written as an encoding")


CASES = [a_cast_of_another_buffer_is_not_read, a_v_i_writes_each_register_of_its_rows,
         a_kernel_vcix_of_no_vcix_form_is_refused]


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
