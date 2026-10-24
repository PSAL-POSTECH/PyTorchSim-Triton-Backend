"""lib_mxu.registers_at_pops, what a dot's stream budgets for an epilogue at its pops: the values its
ops keep live together, the ones it takes from outside and the ones its loops carry, each as wide as
a pop's words in its own element; a splat constant holds no register (it goes to an instruction as a
scalar), and an index is 64 bits.
    python3 tests/passes/dot_register_budget_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lib_mxu as mxu  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

# A row max over C, its mask from an index vector made outside: the walk attention runs at QK's pops.
WALK = """func.func @f(%C: memref<8x64xf32, 1>, %M: memref<8xf32, 1>) {
  %ninf = arith.constant dense<0xFF800000> : vector<16xf32>
  %scale = arith.constant dense<0.5> : vector<16xf32>
  %step = vector.step : vector<16xindex>
  %idx = arith.index_cast %step : vector<16xindex> to vector<16xi32>
  affine.for %r = 0 to 8 {
    %lim = arith.constant dense<3> : vector<16xi32>
    %m = affine.for %c = 0 to 64 step 16 iter_args(%acc = %ninf) -> (vector<16xf32>) {
      %x = affine.vector_load %C[%r, %c] : memref<8x64xf32, 1>, vector<16xf32>
      %s = arith.mulf %x, %scale : vector<16xf32>
      %keep = arith.cmpi slt, IDX, %lim : vector<16xi32>
      %v = arith.select %keep, %s, %ninf : vector<16xi1>, vector<16xf32>
      %n = arith.maximumf %acc, %v : vector<16xf32>
      affine.yield %n : vector<16xf32>
    }
    %red = vector.reduction <maximumf>, %m : vector<16xf32> into f32
    %row = vector.broadcast %red : f32 to vector<1xf32>
    affine.vector_store %row, %M[%r] : memref<8xf32, 1>, vector<1xf32>
  }
  return
}"""


def _registers(words, index="%idx", vlen=256):
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse(WALK.replace("IDX", index))
        loop = next(o for o in walk(m.operation) if o.name == "affine.for")
        return mxu.registers_at_pops(loop, words, vlen)


def a_splat_constant_holds_no_register():
    """The mask's other side a splat (%lim) instead of the index made outside: the index's register
    goes, and %lim adds none."""
    assert _registers(8, index="%lim") == _registers(8) - 1, (_registers(8, index="%lim"), _registers(8))


def an_index_from_outside_holds_a_register_at_the_pop_width():
    """At 8 words an i32 is one register, at 16 two."""
    with_8, without_8 = _registers(8), _registers(8, index="%lim")
    with_16, without_16 = _registers(16), _registers(16, index="%lim")
    assert (with_8 - without_8, with_16 - without_16) == (1, 2), (with_8, without_8, with_16, without_16)


def the_count_grows_with_the_pop_width():
    assert _registers(16) > _registers(8), (_registers(8), _registers(16))


CASES = [a_splat_constant_holds_no_register, an_index_from_outside_holds_a_register_at_the_pop_width,
         the_count_grows_with_the_pop_width]


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
