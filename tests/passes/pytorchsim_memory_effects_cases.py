"""What pytorchsim's ops declare they read and write, asked through upstream passes alone: affine LICM
keeps an access in a loop whose pytorchsim op touches that buffer, and canonicalize/cse never drop or
merge a transfer, dot, copy or barrier. python3 tests/passes/pytorchsim_memory_effects_cases.py"""
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
SEG5 = "operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>"
LICM = "builtin.module(func.func(affine-loop-invariant-code-motion))"


def _transfer(kind, sram, sram_t, extra="", index=None):
    seg, operands, types = SEG5, f"%d, %i, {sram}, %c0, %c0", f"{DRAM}, index, {sram_t}, index, index"
    if index:
        seg = seg.replace("1, 1, 1, 1, 1, 0", "1, 1, 1, 1, 1, 1")
        operands += f", {index}"
        types += f", {BUF}"
        extra += ", offset_stride = 1 : i64"
    res = " -> tensor<64xf32>" if sram_t.startswith("tensor") else " -> ()"
    return (f'"pytorchsim.transfer"({operands}) <{{direction = "{kind}", dram_arg = 0 : i64, '
            f"dram_stride = [1]{extra}, {seg}}}> : ({types}){res}")


def _loop(body):
    return (f"func.func @f(%d: {DRAM}, %m: {BUF}, %s: {BUF}, %o: memref<4x64xf32, 1>, "
            "%x: f32, %a: memref<4x4xf32, 1>, %b: memref<4x4xf32, 1>, %c: memref<4x4xf32, 1>) {\n"
            "  %c0 = arith.constant 0 : index\n"
            "  affine.for %i = 0 to 4 {\n" + body + "  }\n  return\n}\n")


def _licm(text):
    m = ir.Module.parse(text)
    PassManager.parse(LICM).run(m.operation)
    assert m.operation.verify()
    return m


def _in_loop(m, name):
    loop = [o for o in walk(m.operation) if o.name == "affine.for"][0]
    return [o.name for o in walk(loop) if o.name == name]


LOAD_M = (f"    %v = affine.vector_load %m[%c0] : {BUF}, vector<64xf32>\n"
          "    affine.vector_store %v, %o[%i, %c0] : memref<4x64xf32, 1>, vector<64xf32>\n")
STORE_M = f"    affine.store %x, %m[0] : {BUF}\n"


def mvin_into_a_buffer_keeps_the_load_inside():
    m = _licm(_loop("    " + _transfer("load", "%m", BUF) + "\n" + LOAD_M))
    assert _in_loop(m, "affine.vector_load"), m


def mvout_reading_a_buffer_keeps_the_store_inside():
    m = _licm(_loop("    " + _transfer("store", "%m", BUF) + "\n" + STORE_M))
    assert _in_loop(m, "affine.store"), m


def gather_index_read_keeps_the_store_inside():
    m = _licm(_loop("    " + _transfer("load", "%s", BUF, index="%m") + "\n" + STORE_M))
    assert _in_loop(m, "affine.store"), m


def _copy(src, dst):
    return (f'    "pytorchsim.convert_layout"({src}, {dst}) <{{dst_perm = array<i64: 0>, src_perm = array<i64: 0>}}>'
            f" : ({BUF}, {BUF}) -> ()\n")


def copy_into_a_buffer_keeps_the_load_inside():
    m = _licm(_loop(_copy("%s", "%m") + LOAD_M))
    assert _in_loop(m, "affine.vector_load"), m


def copy_reading_a_buffer_lets_its_load_leave():
    m = _licm(_loop(_copy("%m", "%s") + LOAD_M))
    assert not _in_loop(m, "affine.vector_load"), m


def dot_accumulates_into_init_keeps_the_store_inside():
    t = "memref<4x4xf32, 1>"
    m = _licm(_loop(f'    "pytorchsim.dot"(%a, %b, %c) : ({t}, {t}, {t}) -> ()\n'
                    f"    affine.store %x, %c[0, 0] : {t}\n"))
    assert _in_loop(m, "affine.store"), m


def a_fold_into_a_buffer_keeps_the_load_inside():
    m = _licm(_loop(f'    "pytorchsim.distributed_reduction"(%s, %m) <{{by = "arith.addf"}}> '
                    f': ({BUF}, {BUF}) -> ()\n' + LOAD_M))
    assert _in_loop(m, "affine.vector_load"), m


def an_xlu_move_into_a_buffer_keeps_the_load_inside():
    m = _licm(_loop(f'    "pytorchsim_tpu.xlu_move"(%s, %m) <{{mode = "permute"}}>'
                    f' : ({BUF}, {BUF}) -> ()\n' + LOAD_M))
    assert _in_loop(m, "affine.vector_load"), m


def _count(text, pipeline, name):
    m = ir.Module.parse(text)
    PassManager.parse(pipeline).run(m.operation)
    assert m.operation.verify()
    return len([o for o in walk(m.operation) if o.name == name])


def _tensor_func(body, ret="", ret_t=""):
    return (f"func.func @f(%d: {DRAM}, %i: index) -> ({ret_t}) {{\n"
            "  %c0 = arith.constant 0 : index\n"
            "  %e = tensor.empty() : tensor<64xf32>\n" + body + f"  return {ret}\n}}\n")


def unused_tensor_mvin_survives_canonicalize():
    """Pins the identity contract (holds before effects were declared too): its semaphore write."""
    text = _tensor_func("  %t = " + _transfer("load", "%e", "tensor<64xf32>") + "\n")
    assert _count(text, "builtin.module(canonicalize)", "pytorchsim.transfer") == 1


def twin_tensor_mvins_are_not_merged():
    """Pins the identity contract (holds before effects were declared too): two signals, not one."""
    tx = _transfer("load", "%e", "tensor<64xf32>")
    text = _tensor_func(f"  %t = {tx}\n  %u = {tx}\n", "%t, %u : tensor<64xf32>, tensor<64xf32>",
                        "tensor<64xf32>, tensor<64xf32>")
    assert _count(text, "builtin.module(canonicalize,cse)", "pytorchsim.transfer") == 2


def unused_tensor_dot_and_copy_survive():
    """Pins the identity contract (holds before effects were declared too): each occupies a unit."""
    t = "tensor<4x4xf32>"
    body = (f"  %z = tensor.empty() : {t}\n"
            f'  %p = "pytorchsim.dot"(%z, %z, %z) : ({t}, {t}, {t}) -> {t}\n'
            f'  %q = "pytorchsim.dot"(%z, %z, %z) : ({t}, {t}, {t}) -> {t}\n'
            '  %r = "pytorchsim.convert_layout"(%e, %e) <{dst_perm = array<i64: 0>, src_perm = array<i64: 0>}>'
            " : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>\n")
    text = _tensor_func(body)
    assert _count(text, "builtin.module(canonicalize,cse)", "pytorchsim.dot") == 2
    assert _count(text, "builtin.module(canonicalize,cse)", "pytorchsim.convert_layout") == 1


def barrier_ops_survive_greedy():
    """Pins the identity contract (holds before effects were declared too): nothing with no result
    the TOG calls for its effect is erased by a greedy driver."""
    text = ("func.func @f() {\n  %c0 = arith.constant 0 : index\n  %c64 = arith.constant 64 : index\n"
            '  "togsim.dma"(%c0, %c0) : (index, index) -> ()\n'
            '  "togsim.compute"() : () -> ()\n'
            '  "togsim.memory_barrier"(%c0) : (index) -> ()\n'
            '  "pytorchsim.wait"(%c0, %c64) : (index, index) -> ()\n'
            "  return\n}\n")
    pipeline = "builtin.module(canonicalize,cse,convert-vector-to-scf{full-unroll=true},canonicalize)"
    for name in ("togsim.dma", "togsim.compute", "togsim.memory_barrier", "pytorchsim.wait"):
        assert _count(text, pipeline, name) == 1, name


CASES = [mvin_into_a_buffer_keeps_the_load_inside,
         mvout_reading_a_buffer_keeps_the_store_inside,
         gather_index_read_keeps_the_store_inside,
         copy_into_a_buffer_keeps_the_load_inside,
         copy_reading_a_buffer_lets_its_load_leave,
         dot_accumulates_into_init_keeps_the_store_inside,
         a_fold_into_a_buffer_keeps_the_load_inside,
         an_xlu_move_into_a_buffer_keeps_the_load_inside,
         unused_tensor_mvin_survives_canonicalize,
         twin_tensor_mvins_are_not_merged,
         unused_tensor_dot_and_copy_survive,
         barrier_ops_survive_greedy]


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
