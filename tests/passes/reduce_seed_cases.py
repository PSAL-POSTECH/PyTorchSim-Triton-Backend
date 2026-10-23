"""torchsim-vectorize seeds a reduce with the fill just before it in its block (a dealloc elsewhere is no
reader of that init), and leaves a scalar store into a banked scratchpad buffer for its check to refuse.
    python3 tests/passes/reduce_seed_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_runner import run  # noqa: E402
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import p05_vectorize  # noqa: E402

torchsim.register()


def a_fill_seeds_the_reduce_in_its_loop_though_the_buffer_is_freed_at_the_entry():
    """The reduce's init is the fill just before it in the loop's block; the buffer's dealloc stands in
    the entry block, ending its life, and is no reader of the init."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = ir.Module.parse("""module {
  func.func @f(%x: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %t = memref.alloc() : memref<128x64xf32, 1>
    %r = memref.alloc() : memref<128xf32, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      linalg.fill ins(%x : f32) outs(%r : memref<128xf32, 1>)
      linalg.reduce ins(%t : memref<128x64xf32, 1>) outs(%r : memref<128xf32, 1>) dimensions = [1]
        (%in: f32, %acc: f32) {
          %s = arith.addf %in, %acc : f32
          linalg.yield %s : f32
        }
    }
    memref.dealloc %t : memref<128x64xf32, 1>
    memref.dealloc %r : memref<128xf32, 1>
    return
  }
}""")
        seeds = p05_vectorize._find_seeding_fills(m)
    assert len(seeds) == 1 and list(seeds[0][1]) == [1], seeds


def a_scalar_written_into_a_banked_buffer_is_still_refused():
    """A store into a buffer of more than one element addresses it as flat; making it a vector<1>
    would hide that from the check after it, so it is left for the check to refuse."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse("""func.func @f(%x: f32, %a: memref<128xf32, 1>) {
  %c3 = arith.constant 3 : index
  memref.store %x, %a[%c3] : memref<128xf32, 1>
  return
}
""")
        try:
            p05_vectorize.run(m, lanes=128)
        except p05_vectorize.NotVectorisable as e:
            assert "as if it were flat" in str(e), str(e)
            return
    raise AssertionError("not refused")


CASES = [a_fill_seeds_the_reduce_in_its_loop_though_the_buffer_is_freed_at_the_entry,
         a_scalar_written_into_a_banked_buffer_is_still_refused]


def main():
    return run(CASES)


if __name__ == "__main__":
    sys.exit(main())
