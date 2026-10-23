"""What torchsim states about aliasing, asked through an upstream pass alone: a bank view or a pack
is the buffer it is made from, so affine-scalrep's alias analysis follows a store through one to
that buffer and no further.  python3 tests/passes/torchsim_alias_cases.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run  # noqa: E402
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.passmanager import PassManager  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

torchsim.register()

TILE = "memref<4x8xf32, 1>"
BANK = "memref<1x8xf32, strided<[8, 1]>, 1>"
HALF = "memref<1x8xf16, 1>"
WORDS = "memref<1x4xi32, 1>"
ROW = "memref<4xf32, 1>"


def _loads_left(text, pipeline):
    m = ir.Module.parse(text)
    PassManager.parse(pipeline).run(m.operation)
    assert m.operation.verify()
    return len([o for o in walk(m.operation) if o.name == "affine.load"])


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
    write = (f'  %bv = "torchsim.bank_view"(%a) : ({TILE}) -> {BANK}\n'
             f"  affine.store %w, %bv[0, 0] : {BANK}\n")
    assert _loads_left(_forward(write), SCALREP) == 0


def a_store_through_another_buffers_pack_lets_the_load_forward():
    """Breaks if pack stops being a view of the bank it reads as words."""
    write = (f'  %p = "torchsim.pack"(%h) : ({HALF}) -> {WORDS}\n'
             f"  affine.store %k, %p[0, 0] : {WORDS}\n")
    assert _loads_left(_forward(write), SCALREP) == 0


def a_store_through_the_same_buffers_bank_view_keeps_the_load():
    """Holds with or without the view: a store through %b's own bank view may overwrite %b."""
    write = (f'  %bv = "torchsim.bank_view"(%b) : ({ROW}) -> memref<1x4xf32, 1>\n'
             "  affine.store %w, %bv[0, 0] : memref<1x4xf32, 1>\n")
    assert _loads_left(_forward(write), SCALREP) == 1


CASES = [a_store_through_another_buffers_bank_view_lets_the_load_forward,
         a_store_through_another_buffers_pack_lets_the_load_forward,
         a_store_through_the_same_buffers_bank_view_keeps_the_load]


def main():
    return run(CASES, mlir=True)


if __name__ == "__main__":
    sys.exit(main())
