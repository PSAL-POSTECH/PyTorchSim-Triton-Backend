"""TORCHSIM_COMPILE_SPAD_POISON, off by default: a byte every .spad slot starts as in place of zero, in the
ELF's image and refilled by the wrapper in every lane before every program (the image reaches lane 0 only).
    python3 tests/passes/spad_poison_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import torchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import lower, p13_lower_to_llvm as p13  # noqa: E402

torchsim.register()
T = machine.target()

_KINDS = """module {
  func.func @f(%x: f32, %h: bf16, %n: i32, %b: i1) {
    %c0 = arith.constant 0 : index
    %a = memref.alloc() : memref<4xf32, 1>
    %g = memref.alloc() : memref<4xbf16, 1>
    %i = memref.alloc() : memref<4xi32, 1>
    %p = memref.alloc() : memref<4xi1, 1>
    memref.store %x, %a[%c0] : memref<4xf32, 1>
    memref.store %h, %g[%c0] : memref<4xbf16, 1>
    memref.store %n, %i[%c0] : memref<4xi32, 1>
    memref.store %b, %p[%c0] : memref<4xi1, 1>
    memref.dealloc %p : memref<4xi1, 1>
    memref.dealloc %i : memref<4xi32, 1>
    memref.dealloc %g : memref<4xbf16, 1>
    memref.dealloc %a : memref<4xf32, 1>
    return
  }
}"""


class _Poison:
    def __init__(self, said):
        self.said = said

    def __enter__(self):
        self.old = os.environ.get(p13.SPAD_POISON_ENV)
        if self.said is None:
            os.environ.pop(p13.SPAD_POISON_ENV, None)
        else:
            os.environ[p13.SPAD_POISON_ENV] = self.said

    def __exit__(self, *_):
        if self.old is None:
            os.environ.pop(p13.SPAD_POISON_ENV, None)
        else:
            os.environ[p13.SPAD_POISON_ENV] = self.old


def _filled(ety, count, byte):
    """`count` elements of `ety` holding `byte` in every byte, built here from the bits."""
    import numpy as np
    width = ety.width
    bits = int.from_bytes(bytes([byte]) * -(-width // 8), "little") & ((1 << width) - 1)
    if isinstance(ety, ir.IntegerType):
        signed = bits - (1 << width) if bits >> (width - 1) else bits
        return ir.DenseElementsAttr.get_splat(ir.RankedTensorType.get([count], ety),
                                              ir.IntegerAttr.get(ety, signed))
    word = {16: np.uint16, 32: np.uint32, 64: np.uint64}[width]
    return ir.DenseElementsAttr.get(np.full([1], bits, dtype=word), type=ety, shape=[count])


def _check(said, byte):
    """TO_LLVM over _KINDS with the knob at `said`: every slot holds `byte` throughout (0, the zero
    _zero_tile writes, when off); the poisoned extent's value, or None when none is stated."""
    from mlir.dialects.llvm import ArrayType
    with _Poison(said), tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "in.mlir"), os.path.join(d, "out.mlir")
        with open(src, "w") as f:
            f.write(_KINDS)
        lower.to_llvm(src, out, target=T)
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx, ir.Location.unknown(), open(out) as f:
            m = ir.Module.parse(f.read())
            slots, reach = [], None
            for o in m.body.operations:
                o = o.operation
                if o.name != "llvm.mlir.global":
                    continue
                name = ir.StringAttr(o.attributes["sym_name"]).value
                if name == p13.SPAD_POISON_BYTES_SYMBOL:
                    reach = ir.IntegerAttr(o.attributes["value"]).value
                    continue
                if "section" not in o.attributes or \
                        ir.StringAttr(o.attributes["section"]).value != p13.SPAD_SECTION:
                    continue
                ty = ArrayType(ir.TypeAttr(o.attributes["global_type"]).value)
                slots.append(ty.element_type)
                want = _filled(ty.element_type, ty.num_elements, byte)
                assert o.attributes["value"] == want, (name, str(o.attributes["value"]), str(want))
            assert len(slots) == 4, slots
            return reach


def off_by_default_every_slot_is_zero_and_no_extent_is_stated():
    for said in (None, "0", ""):
        assert _check(said, 0) is None, said


def a_poison_byte_fills_every_byte_of_every_element_whatever_its_type():
    """f32, bf16, i32 and i1 slots: 0x41 and 0xff in every byte of each."""
    for byte in (0x41, 0xFF):
        _check(hex(byte), byte)


def the_poisoned_extent_covers_every_slot_at_its_alignment():
    """Four slots of 4 elements, each rounded to the register boundary .spad aligns them at."""
    reach = _check("0xff", 0xFF)
    assert reach == 4 * (T.vlen_bits // 8), reach


def the_extent_takes_the_bank_top_anchors_linkage_whatever_global_comes_first():
    """The extent must link as the wrapper's extern finds it: it takes the linkage of the anchor
    reserve_spad_storage just made, not of whichever llvm.mlir.global happens to be first (here a
    private one). Before, it copied the first global's, so call order decided it."""
    import inspect
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        m = ir.Module.parse("module {\n  llvm.mlir.global private constant @x(0 : i64) : i64\n}")
        top = m.body
        p13._mint_the_bank_top_anchor(top)

        def named(n):
            return next(o.operation for o in top.operations
                        if ir.StringAttr(o.operation.attributes["sym_name"]).value == n)
        anchor, priv = named(p13.SPAD_TOP_SYMBOL), named("x")
        priv.move_before(top.operations[0].operation)
        took = [("a", 4, ir.F32Type.get())], 16, anchor
        if len(inspect.signature(p13._mint_the_poisoned_extent).parameters) == 3:
            took = took[:2]
        p13._mint_the_poisoned_extent(top, *took)
        ext = named(p13.SPAD_POISON_BYTES_SYMBOL)
        assert ext.attributes["linkage"] == anchor.attributes["linkage"], str(ext)
        assert ext.attributes["linkage"] != priv.attributes["linkage"], str(ext)


def a_knob_that_is_not_a_byte_is_refused():
    for said in ("256", "-1", "poison"):
        try:
            with _Poison(said):
                p13.spad_poison()
        except p13.SpadReservationRefused as e:
            assert p13.SPAD_POISON_ENV in str(e), str(e)
        else:
            raise AssertionError(f"{said!r} was taken as a poison byte")


def the_wrapper_refills_every_lane_before_every_program_only_when_poisoned():
    from pytorchsim_triton_compiler.contract.kernel_spec import Arg, IN, KernelSpec
    from pytorchsim_triton_compiler.lowering.elf import wrapper
    spec = KernelSpec(name="k", kernel=None, signature={"x_ptr": "*fp32"}, constexprs={},
                      args=[Arg("x", IN, "float32", (4,))], grid=(2,), reference=None,
                      make_inputs=None)
    with _Poison(None):
        plain = wrapper.generate(spec, T)
    with _Poison("0x7f"):
        poisoned = wrapper.generate(spec, T)
        sampled = wrapper.generate(spec, T, single_program=True)
    assert "poison" not in plain and "poison" not in sampled, plain
    lines = poisoned.split("\n")
    call = next(k for k, line in enumerate(lines) if line.strip().startswith("k("))
    assert lines[call - 1].strip() == "poison_spad();", lines[call - 3:call + 1]
    assert f"lane < {T.vector_lanes}" in poisoned and "0x7f7f7f7f7f7f7f7fULL" in poisoned, poisoned
    assert p13.SPAD_POISON_BYTES_SYMBOL in poisoned, poisoned


CASES = [off_by_default_every_slot_is_zero_and_no_extent_is_stated,
         a_poison_byte_fills_every_byte_of_every_element_whatever_its_type,
         the_poisoned_extent_covers_every_slot_at_its_alignment,
         the_extent_takes_the_bank_top_anchors_linkage_whatever_global_comes_first,
         a_knob_that_is_not_a_byte_is_refused,
         the_wrapper_refills_every_lane_before_every_program_only_when_poisoned]


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
