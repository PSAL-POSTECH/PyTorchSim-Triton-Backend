"""What pytorchsim states about aliasing: a bank view or a pack is the buffer it is made from, so
affine-scalrep follows a store through one to that buffer and no further, and the passes' view test
(lib_ir.view_source) names that buffer.  python3 tests/passes/pytorchsim_alias_cases.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.dialects import affine  # noqa: E402
from mlir.passmanager import PassManager  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import view_source, walk  # noqa: E402

pytorchsim.register()

TILE = "memref<4x8xf32, 1>"
BANK = "memref<1x8xf32, strided<[8, 1]>, 1>"
HALF = "memref<1x8xf16, 1>"
WORDS = "memref<1x4xi32, 1>"
ROW = "memref<4xf32, 1>"


def _loads_left(text, pipeline):
    m = ir.Module.parse(text)
    PassManager.parse(pipeline).run(m.operation)
    assert m.operation.verify()
    return len([o for o in walk(m.operation) if isinstance(o.opview, affine.AffineLoadOp)])


def _forward(write):
    """Store %v to %b, then `write` (a store through some view), then load %b back."""
    return ("func.func @f(%v: f32, %w: f32, %k: i32) -> f32 {\n"
            f"  %a = memref.alloc() : {TILE}\n  %b = memref.alloc() : {ROW}\n"
            f"  %h = memref.alloc() : {HALF}\n"
            f"  affine.store %v, %b[0] : {ROW}\n" + write +
            f"  %x = affine.load %b[0] : {ROW}\n  return %x : f32\n}}\n")


SCALREP = "builtin.module(func.func(affine-scalrep))"


def a_store_through_another_buffers_bank_view_lets_the_load_forward():
    """Breaks if bank_view stops being a view: its result is then a root of its own that may alias
    %b, and the store through it blocks forwarding %v to the load."""
    write = (f'  %bv = "pytorchsim_tpu.vpu.bank_view"(%a) : ({TILE}) -> {BANK}\n'
             f"  affine.store %w, %bv[0, 0] : {BANK}\n")
    assert _loads_left(_forward(write), SCALREP) == 0


def a_store_through_another_buffers_pack_lets_the_load_forward():
    """Breaks if pack stops being a view of the bank it reads as words."""
    write = (f'  %p = "pytorchsim_tpu.vpu.bank_words"(%h) : ({HALF}) -> {WORDS}\n'
             f"  affine.store %k, %p[0, 0] : {WORDS}\n")
    assert _loads_left(_forward(write), SCALREP) == 0


def a_store_through_the_same_buffers_bank_view_keeps_the_load():
    """Holds with or without the view: a store through %b's own bank view may overwrite %b."""
    write = (f'  %bv = "pytorchsim_tpu.vpu.bank_view"(%b) : ({ROW}) -> memref<1x4xf32, 1>\n'
             "  affine.store %w, %bv[0, 0] : memref<1x4xf32, 1>\n")
    assert _loads_left(_forward(write), SCALREP) == 1


def _sources(body, args=""):
    """view_source of each op in `body` (a function of `args`), both by their printed names."""
    m = ir.Module.parse(f"func.func @f({args}) {{\n{body}  return\n}}\n")
    func = m.body.operations[0]
    out = {}
    for o in walk(func.operation):
        if o.results:
            src = view_source(o)
            out[o.results[0].get_name()] = None if src is None else src.get_name()
    return out


def a_bank_view_and_bank_words_are_views_of_their_source():
    """Breaks if the passes' view test is memref's ops alone: a reader through a bank view is then
    not a reader of its buffer, and the TOG keeps the view as compute."""
    got = _sources(f'  %a = memref.alloc() : {TILE}\n'
                   f'  %bv = "pytorchsim_tpu.vpu.bank_view"(%a) : ({TILE}) -> {BANK}\n'
                   f'  %h = memref.alloc() : {HALF}\n'
                   f'  %p = "pytorchsim_tpu.vpu.bank_words"(%h) : ({HALF}) -> {WORDS}\n')
    assert got == {"%alloc": None, "%0": "%alloc", "%alloc_0": None, "%1": "%alloc_0"}, got


def a_view_is_one_in_its_source_space_whatever_op_makes_it():
    """memref.view and a reinterpret_cast of an unranked argument are views; a cast into another
    memory space names another buffer, not a view of this one."""
    got = _sources("  %c0 = arith.constant 0 : index\n"
                   "  %v = memref.view %bytes[%c0][] : memref<64xi8, 1> to memref<16xf32, 1>\n"
                   "  %r = memref.reinterpret_cast %u to offset: [0], sizes: [8], strides: [1]"
                   " : memref<*xf32> to memref<8xf32>\n"
                   "  %s = memref.memory_space_cast %r : memref<8xf32> to memref<8xf32, 1>\n",
                   "%bytes: memref<64xi8, 1>, %u: memref<*xf32>")
    assert got == {"%c0": None, "%view": "%arg0", "%reinterpret_cast": "%arg1", "%memspacecast": None}, got


CASES = [a_store_through_another_buffers_bank_view_lets_the_load_forward,
         a_store_through_another_buffers_pack_lets_the_load_forward,
         a_store_through_the_same_buffers_bank_view_keeps_the_load,
         a_bank_view_and_bank_words_are_views_of_their_source,
         a_view_is_one_in_its_source_space_whatever_op_makes_it]


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
