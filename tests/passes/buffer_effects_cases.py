"""What an op does to a buffer is read from the effects it states (effects_on), never its name: a
free ends the buffer's life and neither reads nor writes it, a wait's landing is the state of the
transfers into it and no data, so no walker that asks who reads or writes a buffer sees either.
    python3 tests/passes/buffer_effects_cases.py        exit 0 iff every case holds"""
import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run  # noqa: E402
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.passmanager import PassManager  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lib_ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lib_mxu  # noqa: E402

torchsim.register()

BUF = "memref<8x16xf32, 1>"
_Use = namedtuple("_Use", "operand_number")


def _module(body):
    return ir.Module.parse("module {\n" + body + "\n}\n")


def _op(m, name):
    return next(o for o in walk(m.operation) if o.name == name)


def _neither(o, k=0):
    return (not lib_ir.writes_through_this_operand(o, _Use(k))
            and not lib_ir.reads_through_this_operand(o, _Use(k)))


def a_dealloc_ends_a_life_and_neither_reads_nor_writes():
    """Breaks if lib_ir classifies memref.dealloc by its fallthrough (a writer, so also a reader)."""
    m = _module(f"func.func @f() {{\n  %a = memref.alloc() : {BUF}\n"
                f"  memref.dealloc %a : {BUF}\n  return\n}}")
    free = _op(m, "memref.dealloc")
    assert torchsim.effects_on(free, free.operands[0]) == {"free"}
    assert lib_ir.ends_lifetime(free, free.operands[0])
    assert _neither(free)
    alloc = _op(m, "memref.alloc")
    assert torchsim.effects_on(alloc, alloc.results[0]) == {"allocate"}


def a_wait_landing_is_neither_read_nor_written():
    """Breaks if torchsim.wait loses `landing`, or its effect on it moves onto the buffer's data."""
    m = _module(f"func.func @f(%s: {BUF}) {{\n  %c0 = arith.constant 0 : index\n"
                f"  %c8 = arith.constant 8 : index\n"
                f'  "torchsim.wait"(%c0, %c8, %s) : (index, index, {BUF}) -> ()\n  return\n}}')
    w = _op(m, torchsim.WaitOp.OPERATION_NAME)
    assert list(w.opview.landing) == [w.operands[2]]
    assert torchsim.effects_on(w, w.operands[2]) == {"sync"}
    assert not lib_ir.ends_lifetime(w, w.operands[2])
    assert _neither(w, 2)


def an_op_stating_no_effects_is_unknown_and_keeps_its_old_answer():
    """Breaks if effects_on guesses for an op without MemoryEffectOpInterface."""
    m = _module(f"func.func @f(%s: {BUF}) {{\n"
                f'  "test.op"(%s) : ({BUF}) -> ()\n  return\n}}')
    o = _op(m, "test.op")
    assert torchsim.effects_on(o, o.operands[0]) == torchsim.UNKNOWN
    assert lib_ir.writes_through_this_operand(o, _Use(0))


def a_view_root_is_the_buffer_in_the_same_space():
    """Breaks if view_root stops short of a view chain's buffer or crosses a space cast."""
    m = _module(f"func.func @f(%d: memref<8x16xf32>) {{\n  %a = memref.alloc() : {BUF}\n"
                f"  %v = memref.subview %a[0, 0] [4, 16] [1, 1] : {BUF} to "
                f"memref<4x16xf32, strided<[16, 1]>, 1>\n"
                f"  %c = memref.memory_space_cast %d : memref<8x16xf32> to {BUF}\n"
                f"  return\n}}")
    alloc, sub = _op(m, "memref.alloc"), _op(m, "memref.subview")
    cast = _op(m, "memref.memory_space_cast")
    assert torchsim.view_root(sub.results[0]) == alloc.results[0]
    assert torchsim.view_root(cast.results[0]) == cast.results[0]


def a_dealloc_stays_after_the_wait_landing_in_its_buffer():
    """Breaks if a wait stops naming its buffer: optimize-allocation-liveness then pulls the free
    up to the last read, ahead of the DMA it still waits on."""
    m = _module(f"func.func @f(%d: memref<128xf32>) -> f32 {{\n  %c0 = arith.constant 0 : index\n"
                f"  %c64 = arith.constant 64 : index\n  %a = memref.alloc() : {BUF}\n"
                f'  "torchsim.transfer"(%d, %c0, %a, %c0, %c0) <{{async = true, dma_kind = "MVOUT", '
                f"dram_arg = 0 : i64, dram_stride = [16, 1], "
                f"operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}> "
                f": (memref<128xf32>, index, {BUF}, index, index) -> ()\n"
                f"  %x = memref.load %a[%c0, %c0] : {BUF}\n"
                f'  "torchsim.wait"(%c0, %c64, %a) : (index, index, {BUF}) -> ()\n'
                f"  memref.dealloc %a : {BUF}\n  return %x : f32\n}}")
    PassManager.parse("builtin.module(func.func(optimize-allocation-liveness))").run(m.operation)
    names = [o.name for o in walk(m.operation)]
    assert names.index("memref.dealloc") > names.index(torchsim.WaitOp.OPERATION_NAME), names


_PRODUCT = (f"func.func @f(%a: memref<4x4xf32, 1>, %b: memref<4x4xf32, 1>) {{\n"
            f"  %c0 = arith.constant 0 : index\n  %c16 = arith.constant 16 : index\n"
            f"  %z = arith.constant 0.0 : f32\n"
            f"  %c = memref.alloc() : memref<4x4xf32, 1>\n"
            f"  linalg.fill ins(%z : f32) outs(%c : memref<4x4xf32, 1>)\n"
            f'  "torchsim.wait"(%c0, %c16, %c) : (index, index, memref<4x4xf32, 1>) -> ()\n'
            f'  "torchsim.dot"(%a, %b, %c) : (memref<4x4xf32, 1>, memref<4x4xf32, 1>, '
            f"memref<4x4xf32, 1>) -> ()\n  return\n}}")


def a_wait_between_the_writer_and_the_dot_is_not_its_last_writer():
    """Breaks if lib_mxu's walkers count a wait's landing as a write or a use of the buffer."""
    m = _module(_PRODUCT)
    dot = _op(m, "torchsim.dot")
    fill, wait = _op(m, "linalg.fill"), _op(m, torchsim.WaitOp.OPERATION_NAME)
    c = dot.operands[2]
    assert lib_mxu.last_writer(dot, c) == fill
    assert lib_mxu.used_between(c, fill, dot) is None
    assert lib_mxu.used_between(c, wait, dot, first=0) is None


def an_epilogue_wait_built_again_keeps_its_landing():
    """Breaks if lib_mxu.build_waits re-mints a moved wait from its first two operands."""
    m = _module(_PRODUCT)
    wait, dot = _op(m, torchsim.WaitOp.OPERATION_NAME), _op(m, "torchsim.dot")
    lib_mxu.build_waits(ir.InsertionPoint(dot), namedtuple("E", "waits")([wait]))
    waits = [o for o in walk(m.operation) if o.name == torchsim.WaitOp.OPERATION_NAME]
    assert len(waits) == 2 and list(waits[1].opview.landing) == [dot.operands[2]], waits


CASES = [a_dealloc_ends_a_life_and_neither_reads_nor_writes,
         a_wait_landing_is_neither_read_nor_written,
         an_op_stating_no_effects_is_unknown_and_keeps_its_old_answer,
         a_view_root_is_the_buffer_in_the_same_space,
         a_dealloc_stays_after_the_wait_landing_in_its_buffer,
         a_wait_between_the_writer_and_the_dot_is_not_its_last_writer,
         an_epilogue_wait_built_again_keeps_its_landing]


def main():
    return run(CASES, unregistered=True)


if __name__ == "__main__":
    sys.exit(main())
