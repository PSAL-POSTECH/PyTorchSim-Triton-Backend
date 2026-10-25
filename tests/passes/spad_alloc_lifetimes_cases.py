"""A scratchpad buffer is a space-1 memref.alloc from bufferize to LLVM lowering, freed at its function's
end; pytorchsim-materialize-spad-allocs makes each a .spad global.
    python3 tests/passes/spad_alloc_lifetimes_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower, a06_bufferize  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

pytorchsim.register()
T = machine.target()

_TWO = """module {
  func.func @f(%x: f32, %d: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %b = memref.alloc() : memref<8x4xf32, 1>
    %a = memref.alloc() : memref<4xf32, 1>
    %vx = vector.broadcast %x : f32 to vector<1xf32>
    vector.store %vx, %a[%c0] : memref<4xf32, 1>, vector<1xf32>
    vector.store %vx, %b[%c0, %c0] : memref<8x4xf32, 1>, vector<1xf32>
    %va = memref.load %a[%c0] : memref<4xf32, 1>
    %vb = memref.load %b[%c0, %c0] : memref<8x4xf32, 1>
    %s = arith.addf %va, %vb : f32
    memref.store %s, %d[%c0] : memref<4xf32>
    memref.dealloc %b : memref<8x4xf32, 1>
    memref.dealloc %a : memref<4xf32, 1>
    return
  }
}"""


def _ctx():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ctx


def a_scratchpad_alloc_reaching_llvm_lowering_is_a_spad_global_not_a_malloc():
    """Two allocs left to TO_LLVM: finalize-memref-to-llvm alone would make each a malloc on the heap.
    Each must instead be a global in .spad, and no malloc or free be declared."""
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "in.mlir"), os.path.join(d, "out.mlir")
        with open(src, "w") as f:
            f.write(_TWO)
        lower.to_llvm(src, out, target=T)
        with _ctx(), open(out) as f:
            m = ir.Module.parse(f.read())
            tops = [o.operation for o in m.body.operations]
            funcs = [ir.StringAttr(o.attributes["sym_name"]).value for o in tops if o.name == "llvm.func"]
            spad = [o for o in tops if o.name == "llvm.mlir.global" and "section" in o.attributes
                    and ir.StringAttr(o.attributes["section"]).value == ".spad"]
    assert not {"malloc", "free"} & set(funcs), funcs
    assert len(spad) == 2, [str(o) for o in tops]


def materialize_keeps_the_order_p04_once_gave_the_globals():
    """a06 moves each alloc ahead of the last, so the entry holds the last made first, and its globals
    were inserted at the module's top: the module order is the entry order, the first made spad0."""
    from pytorchsim_triton_compiler.lowering.passes import c24_materialize_spad_allocs as mat
    with _ctx(), ir.Location.unknown():
        m = ir.Module.parse(_TWO)
        mat.run(m)
        assert m.operation.verify()
        ops = list(walk(m.operation))
        globals_ = [(ir.StringAttr(o.attributes["sym_name"]).value,
                     str(ir.TypeAttr(o.attributes["type"]).value)) for o in ops if o.name == "memref.global"]
        named = [ir.FlatSymbolRefAttr(o.attributes["name"]).value for o in ops if o.name == "memref.get_global"]
        left = [o.name for o in ops if o.name in ("memref.alloc", "memref.dealloc")]
    assert globals_ == [("spad1", "memref<8x4xf32, 1>"), ("spad0", "memref<4xf32, 1>")], globals_
    assert named == ["spad1", "spad0"], named
    assert not left, left


def an_alloc_in_a_loop_moves_to_the_entry_and_is_freed_once_before_the_return():
    with _ctx(), ir.Location.unknown():
        m = ir.Module.parse("""module {
  func.func @f(%x: f32, %d: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %a = memref.alloc() : memref<64xf32>
      memref.store %x, %a[%i] : memref<64xf32>
      %v = memref.load %a[%i] : memref<64xf32>
      memref.store %v, %d[%i] : memref<64xf32>
    }
    return
  }
}""")
        a06_bufferize.run(m)
        assert m.operation.verify()
        fn = [o for o in walk(m.operation) if o.name == "func.func"][0]
        entry = [o.operation for o in fn.regions[0].blocks[0].operations]
        allocs = [o for o in walk(m.operation) if o.name == "memref.alloc"]
        frees = [o for o in walk(m.operation) if o.name == "memref.dealloc"]
        assert len(allocs) == 1 and allocs[0] in entry, [str(o) for o in entry]
        assert ir.IntegerAttr(ir.MemRefType(allocs[0].results[0].type).memory_space).value == 1
        assert len(frees) == 1 and entry.index(frees[0]) == len(entry) - 2, [str(o) for o in entry]


def a_buffer_freed_under_an_if_is_refused_by_p04():
    """A free a06 meets before it gives the buffer its life is left where it stands; under an scf.if it
    would end the buffer on one path only. Refused, never dropped."""
    with _ctx(), ir.Location.unknown():
        m = ir.Module.parse("""module {
  func.func @f(%x: f32, %c: i1, %d: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %a = memref.alloc() : memref<64xf32>
    memref.store %x, %a[%c0] : memref<64xf32>
    %v = memref.load %a[%c0] : memref<64xf32>
    memref.store %v, %d[%c0] : memref<64xf32>
    scf.if %c {
      memref.dealloc %a : memref<64xf32>
    }
    return
  }
}""")
        try:
            a06_bufferize.run(m)
        except a06_bufferize._Fatal as e:
            assert "free" in str(e), str(e)
            return
    raise AssertionError("a dealloc under scf.if passed a06")


def the_static_life_check_refuses_a_conditional_dealloc_and_takes_an_unconditional_one():
    head = """module {
  func.func @f(%x: f32, %c: i1) {
    %c0 = arith.constant 0 : index
    %a = memref.alloc() : memref<4xf32, 1>
    memref.store %x, %a[%c0] : memref<4xf32, 1>
"""
    with _ctx(), ir.Location.unknown():
        ok = ir.Module.parse(head + "    memref.dealloc %a : memref<4xf32, 1>\n    return\n  }\n}")
        a06_bufferize._check_static_lifetimes(ok, a06_bufferize._get_symbols(ok))
        bad = ir.Module.parse(head + "    scf.if %c {\n      memref.dealloc %a : memref<4xf32, 1>\n"
                                     "    }\n    return\n  }\n}")
        try:
            a06_bufferize._check_static_lifetimes(bad, a06_bufferize._get_symbols(bad))
        except a06_bufferize._Fatal as e:
            assert "scf.if" in str(e), str(e)
        else:
            raise AssertionError("a dealloc under scf.if passed the static life check")
        helper = ir.Module.parse(head + "    memref.dealloc %a : memref<4xf32, 1>\n    return\n  }\n"
                                        "  func.func private @dealloc_helper()\n}")
        try:
            a06_bufferize._check_static_lifetimes(helper, {"f"})
        except a06_bufferize._Fatal as e:
            assert "@dealloc_helper" in str(e), str(e)
        else:
            raise AssertionError("a runtime dealloc helper passed the static life check")


CASES = [a_scratchpad_alloc_reaching_llvm_lowering_is_a_spad_global_not_a_malloc,
         materialize_keeps_the_order_p04_once_gave_the_globals,
         an_alloc_in_a_loop_moves_to_the_entry_and_is_freed_once_before_the_return,
         a_buffer_freed_under_an_if_is_refused_by_p04,
         the_static_life_check_refuses_a_conditional_dealloc_and_takes_an_unconditional_one]


def main():
    bad = 0
    for case in CASES:
        try:
            case()
            print(f"ok    {case.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
