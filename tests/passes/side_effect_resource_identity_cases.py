"""Upstream cse keeps two reads of a buffer apart when a torchsim op between them writes it, and the
reverse. Fails until the LLVM fork compares side-effect resources by TypeID (upstream issue 219579):
_pytorchsim_ext holds its own DefaultResource. python3 tests/passes/side_effect_resource_identity_cases.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.passmanager import PassManager  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

pytorchsim.register()

BUF = "memref<64xf32, 1>"
DRAM = "memref<4096xf32>"
T44 = "memref<4x4xf32, 1>"
SEG5 = "operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>"
LOAD = f"memref.load %m[%c0] : {BUF}"
STRIDED_LOAD = (f'"pytorchsim.strided_load"(%m, %c0) <{{count = 4 : i64, stride = 1 : i64}}>'
                f" : ({BUF}, index) -> vector<4xf32>")


def _reads_left(read, writer):
    """How many of two identical `read`s with `writer` between them survive cse."""
    rt = "vector<4xf32>" if "strided_load" in read else "f32"
    text = (f"func.func @f(%d: {DRAM}, %i: index, %m: {BUF}, %s: {BUF}, %x: f32, %v: vector<4xf32>, "
            f"%a: {T44}, %b: {T44}, %c: {T44}) -> ({rt}, {rt}) {{\n"
            "  %c0 = arith.constant 0 : index\n  %c64 = arith.constant 64 : index\n"
            f"  %r0 = {read}\n  {writer}\n  %r1 = {read}\n  return %r0, %r1 : {rt}, {rt}\n}}\n")
    m = ir.Module.parse(text)
    PassManager.parse("builtin.module(cse)").run(m.operation)
    assert m.operation.verify()
    name = "pytorchsim.strided_load" if "strided_load" in read else "memref.load"
    return len([o for o in walk(m.operation) if o.name == name])


def an_upstream_store_keeps_two_loads_apart():
    """The control: writer and readers built in one library; holds before and after the fork bump."""
    assert _reads_left(LOAD, f"memref.store %x, %m[%c0] : {BUF}") == 2


def a_wait_keeps_two_loads_apart():
    assert _reads_left(LOAD, '"pytorchsim.wait"(%c0, %c64) : (index, index) -> ()') == 2


def an_mvin_into_the_buffer_keeps_two_loads_apart():
    mvin = (f'"pytorchsim.transfer"(%d, %i, %m, %c0, %c0) <{{direction = "load", dram_arg = 0 : i64, '
            f"dram_stride = [1], {SEG5}}}> : ({DRAM}, index, {BUF}, index, index) -> ()")
    assert _reads_left(LOAD, mvin) == 2


def a_copy_into_the_buffer_keeps_two_loads_apart():
    copy = (f'"pytorchsim.convert_layout"(%s, %m) <{{dst_perm = array<i64: 0>, src_perm = array<i64: 0>}}>'
            f" : ({BUF}, {BUF}) -> ()")
    assert _reads_left(LOAD, copy) == 2


def a_fold_into_the_buffer_keeps_two_loads_apart():
    assert _reads_left(LOAD, f'"pytorchsim.fold"(%s, %m) <{{by = "arith.addf"}}> : ({BUF}, {BUF}) -> ()') == 2


def an_xlu_move_into_the_buffer_keeps_two_loads_apart():
    assert _reads_left(LOAD, f'"pytorchsim_tpu.xlu_move"(%s, %m) <{{xlu_op = 16 : i64, kind = "permute"}}>'
                       f' : ({BUF}, {BUF}) -> ()') == 2


def a_dot_into_init_keeps_two_loads_apart():
    load = f"memref.load %c[%c0, %c0] : {T44}"
    assert _reads_left(load, f'"pytorchsim.dot"(%a, %b, %c) : ({T44}, {T44}, {T44}) -> ()') == 2


def an_upstream_store_keeps_two_strided_loads_apart():
    assert _reads_left(STRIDED_LOAD, f"memref.store %x, %m[%c0] : {BUF}") == 2


CASES = [an_upstream_store_keeps_two_loads_apart,
         a_wait_keeps_two_loads_apart,
         an_mvin_into_the_buffer_keeps_two_loads_apart,
         a_copy_into_the_buffer_keeps_two_loads_apart,
         a_fold_into_the_buffer_keeps_two_loads_apart,
         an_xlu_move_into_the_buffer_keeps_two_loads_apart,
         a_dot_into_init_keeps_two_loads_apart,
         an_upstream_store_keeps_two_strided_loads_apart]


def main():
    failed = 0
    for case in CASES:
        try:
            with ir.Context(), ir.Location.unknown():
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
