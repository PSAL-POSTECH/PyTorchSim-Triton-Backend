"""PYTORCHSIM_COMPILE_SPAD_POISON, off by default: a byte every .spad slot starts as in place of zero, in the
ELF's image and refilled by the wrapper in every lane before every program (the image reaches lane 0 only);
the lowered module alone says whether it is on, and gem5's timed binary never is; exits 0 iff every case holds."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler import machine, toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402
from pytorchsim_triton_compiler.lowering.passes import (  # noqa: E402
    c24_convert_pytorchsim_to_llvm as c24, lower)

pytorchsim.register()
T = machine.target()

_KINDS = """module {
  func.func @f(%x: f32, %h: bf16, %n: i32, %b: i1) {
    %c0 = arith.constant 0 : index
    %a = memref.alloc() : memref<4xf32, 1>
    %g = memref.alloc() : memref<4xbf16, 1>
    %i = memref.alloc() : memref<4xi32, 1>
    %p = memref.alloc() : memref<4xi1, 1>
    %vx = vector.broadcast %x : f32 to vector<1xf32>
    %vh = vector.broadcast %h : bf16 to vector<1xbf16>
    %vn = vector.broadcast %n : i32 to vector<1xi32>
    vector.store %vx, %a[%c0] : memref<4xf32, 1>, vector<1xf32>
    vector.store %vh, %g[%c0] : memref<4xbf16, 1>, vector<1xbf16>
    vector.store %vn, %i[%c0] : memref<4xi32, 1>, vector<1xi32>
    %l = memref.load %p[%c0] : memref<4xi1, 1>
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
        self.old = os.environ.get(c24.SPAD_POISON_ENV)
        if self.said is None:
            os.environ.pop(c24.SPAD_POISON_ENV, None)
        else:
            os.environ[c24.SPAD_POISON_ENV] = self.said

    def __exit__(self, *_):
        if self.old is None:
            os.environ.pop(c24.SPAD_POISON_ENV, None)
        else:
            os.environ[c24.SPAD_POISON_ENV] = self.old


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


def _lower(said, d, timing=False):
    """_KINDS through TO_LLVM with the knob at `said`, into `d`; the lowered file."""
    src, out = os.path.join(d, "in.mlir"), os.path.join(d, "out.mlir")
    with _Poison(said):
        with open(src, "w") as f:
            f.write(_KINDS)
        lower.to_llvm(src, out, target=T, timing=timing)
    return out


def _check(said, byte, timing=False):
    """TO_LLVM over _KINDS with the knob at `said`: every slot holds `byte` throughout (0, the zero
    _zero_tile writes, when off); the poisoned extent's value, or None when none is stated."""
    from mlir.dialects.llvm import ArrayType
    with tempfile.TemporaryDirectory() as d:
        out = _lower(said, d, timing)
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
                if name == c24.SPAD_POISON_BYTES_SYMBOL:
                    reach = ir.IntegerAttr(o.attributes["value"]).value
                    continue
                if name == c24.SPAD_POISON_WORD_SYMBOL:
                    word = ir.IntegerAttr(o.attributes["value"]).value & (1 << 64) - 1
                    assert word == int.from_bytes(bytes([byte]) * 8, "little"), hex(word)
                    continue
                if "section" not in o.attributes or \
                        ir.StringAttr(o.attributes["section"]).value != c24.SPAD_SECTION:
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
        c24._mint_the_bank_top_anchor(top)

        def named(n):
            return next(o.operation for o in top.operations
                        if ir.StringAttr(o.operation.attributes["sym_name"]).value == n)
        anchor, priv = named(c24.SPAD_TOP_SYMBOL), named("x")
        priv.move_before(top.operations[0].operation)
        took = [("a", 4, ir.F32Type.get())], 16, anchor
        if len(inspect.signature(c24._mint_the_poisoned_extent).parameters) == 3:
            took = took[:2]
        c24._mint_the_poisoned_extent(top, *took)
        ext = named(c24.SPAD_POISON_BYTES_SYMBOL)
        assert ext.attributes["linkage"] == anchor.attributes["linkage"], str(ext)
        assert ext.attributes["linkage"] != priv.attributes["linkage"], str(ext)


def a_knob_that_is_not_a_byte_is_refused():
    for said in ("256", "-1", "poison"):
        try:
            with _Poison(said):
                c24.spad_poison()
        except c24.SpadReservationRefused as e:
            assert c24.SPAD_POISON_ENV in str(e), str(e)
        else:
            raise AssertionError(f"{said!r} was taken as a poison byte")


def _spec():
    from pytorchsim_triton_compiler.contract.kernel_spec import Arg, IN, KernelSpec
    return KernelSpec(name="k", kernel=None, signature={"x_ptr": "*fp32"}, constexprs={},
                      args=[Arg("x", IN, "float32", (4,))], grid=(2,), reference=None,
                      make_inputs=None)


def the_wrapper_refills_every_lane_before_every_program_only_when_poisoned():
    """Whether to refill is the `poisoned` it is handed, never the environment; the word and the
    extent are the lowered module's symbols, strong, so a module without them does not link."""
    from pytorchsim_triton_compiler.lowering.elf import wrapper
    with _Poison("0x7f"):
        plain = wrapper.generate(_spec(), T)
    with _Poison(None):
        poisoned = wrapper.generate(_spec(), T, poisoned=True)
    assert "poison" not in plain, plain
    lines = poisoned.split("\n")
    call = next(k for k, line in enumerate(lines) if line.strip().startswith("k("))
    assert lines[call - 1].strip() == "poison_spad();", lines[call - 3:call + 1]
    assert f"lane < {T.vector_lanes}" in poisoned and "weak" not in poisoned, poisoned
    for sym in (c24.SPAD_POISON_BYTES_SYMBOL, c24.SPAD_POISON_WORD_SYMBOL):
        assert f"extern const" in poisoned and sym in poisoned, poisoned


def whether_a_module_is_poisoned_is_read_off_the_module_in_either_order():
    """The reviewer's --from binary: lowered without the knob, built with it, the wrapper emitted
    poison_spad() and the weak extent made it do nothing. Now the build reads c24.poisoned(lowered):
    lowered plain is plain whatever the build's environment, lowered poisoned is poisoned without it."""
    with tempfile.TemporaryDirectory() as d:
        out = _lower(None, d)
        with _Poison("0xff"):
            assert not c24.poisoned(out)
    with tempfile.TemporaryDirectory() as d:
        out = _lower("0xff", d)
        with _Poison(None):
            assert c24.poisoned(out)


def the_timed_binary_is_never_poisoned():
    """gem5's one-tile binary times, and no one reads its data: TO_LLVM with timing=True reserves
    zeros and states no extent whatever the knob says, so its wrapper refills nothing."""
    assert _check("0xff", 0, timing=True) is None


def a_hand_written_wrapper_over_a_poisoned_module_is_refused():
    """A *_wrapper.c override refills nothing: over a poisoned module lane 0 would start poisoned and
    every other lane zero. Refused, never silently half-poisoned; over a plain module it is used."""
    from pytorchsim_triton_compiler.lowering.elf import build
    from pytorchsim_triton_compiler.stage import StageError

    class Ctx:
        def __init__(self, d):
            self.kernel_path, self.spec, self.said = os.path.join(d, "k.py"), _spec(), []

        def log(self, msg):
            self.said.append(msg)
    for said, refused in ((None, False), ("0xff", True)):
        with tempfile.TemporaryDirectory() as d:
            out = _lower(said, d)
            with open(os.path.join(d, "k_wrapper.c"), "w") as f:
                f.write("int main(void) { return 0; }\n")
            ctx, wrap = Ctx(d), os.path.join(d, "w.c")
            try:
                build._write_wrapper(ctx, "binary", "stage6", out, wrap, False)
            except StageError as e:
                assert refused and c24.SPAD_POISON_ENV in str(e), str(e)
            else:
                assert not refused and os.path.exists(wrap), said


CASES = [off_by_default_every_slot_is_zero_and_no_extent_is_stated,
         a_poison_byte_fills_every_byte_of_every_element_whatever_its_type,
         the_poisoned_extent_covers_every_slot_at_its_alignment,
         the_extent_takes_the_bank_top_anchors_linkage_whatever_global_comes_first,
         a_knob_that_is_not_a_byte_is_refused,
         the_wrapper_refills_every_lane_before_every_program_only_when_poisoned,
         whether_a_module_is_poisoned_is_read_off_the_module_in_either_order,
         the_timed_binary_is_never_poisoned,
         a_hand_written_wrapper_over_a_poisoned_module_is_refused]


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
