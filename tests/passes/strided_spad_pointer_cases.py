"""c24's lower_strided_accesses: a strided access to a space-1 memref goes through a !llvm.ptr<1>,
so "a scratchpad access is an address-space-1 access" holds for vp.strided too; others stay space 0.
    python3 tests/passes/strided_spad_pointer_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from mlir.dialects import llvm  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import c24_convert_pytorchsim_to_llvm as c24  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_bank_layout import SPAD_SPACE  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402


def _spaces(space):
    mem = f"memref<128x8xf32, strided<[16, 2]>{', ' + str(space) if space else ''}>"
    text = (f"func.func @f(%m: {mem}, %i: index, %j: index) {{\n"
            f"  %v = \"pytorchsim.strided_load\"(%m, %i, %j) <{{count = 8 : i64, stride = 2 : i64}}> : "
            f"({mem}, index, index) -> vector<8xf32>\n"
            f"  \"pytorchsim.strided_store\"(%v, %m, %i, %j) <{{count = 8 : i64, stride = 2 : i64}}> : "
            f"(vector<8xf32>, {mem}, index, index) -> ()\n"
            "  return\n}\n")
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse(text)
        c24.lower_strided_accesses(m)
        made = [o for o in walk(m.operation) if isinstance(o.opview, llvm.IntToPtrOp)]
        return [llvm.PointerType(o.results[0].type).address_space for o in made]


def a_scratchpad_strided_access_takes_a_space_1_pointer():
    assert _spaces(SPAD_SPACE) == [SPAD_SPACE, SPAD_SPACE], _spaces(SPAD_SPACE)


def a_strided_access_elsewhere_keeps_space_0():
    assert _spaces(0) == [0, 0], _spaces(0)


CASES = [a_scratchpad_strided_access_takes_a_space_1_pointer,
         a_strided_access_elsewhere_keeps_space_0]

if __name__ == "__main__":
    bad = 0
    for c in CASES:
        try:
            c()
            print(f"ok   {c.__name__}")
        except Exception as e:
            bad += 1
            print(f"FAIL {c.__name__}: {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1 if bad else 0)
