"""The `cut` digit records: what the transfer verifier refuses, and what renumbered_per_axis states
when it collapses or permutes cut axes -- the joined digits address every coordinate as before.
    python3 tests/passes/cut_digits_cases.py        exit 0 iff every case holds"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.lowering.passes import lib_ir  # noqa: E402
from pytorchsim_triton_compiler.dialects import pytorchsim  # noqa: E402

_MOD = """
module {{
  memref.global "private" @spad : memref<{shape}xf32, 1>
  func.func @f(%x: memref<65536xf32>) {{
    %c0 = arith.constant 0 : index
    %s = memref.get_global @spad : memref<{shape}xf32, 1>
    "pytorchsim.transfer"(%x, %c0, %s, %c0, %c0) <{{dma_kind = "MVIN", dram_arg = 0 : i64,
        dram_stride = {dstride}, cut = [{cut}],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 0>}}>
        : (memref<65536xf32>, index, memref<{shape}xf32, 1>, index, index) -> ()
    return
  }}
}}
"""


def _transfer(shape, dstride, cut):
    """The one transfer of a module whose tile is `shape`, cut per axis by `cut` (flat records)."""
    from mlir.ir import Module
    entries = ", ".join(f"array<i64: {', '.join(map(str, e))}>" if e else "array<i64>" for e in cut)
    module = Module.parse(_MOD.format(shape="x".join(map(str, shape)), dstride=list(dstride),
                                      cut=entries))
    return module, next(o for o in lib_ir.walk(module.operation)
                        if o.name == pytorchsim.TransferOp.OPERATION_NAME)


def _refused(shape, dstride, cut, words):
    try:
        _transfer(shape, dstride, cut)
    except Exception as e:  # noqa: BLE001
        assert words in str(e), e
        return
    raise AssertionError(f"the verifier took cut {cut} on dram_stride {dstride}")


def _entries(attrs):
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr
    return [list(DenseI64ArrayAttr(a)) for a in ArrayAttr(attrs["cut"])] if "cut" in attrs else None


def _address(shape, dstride, cut, coord):
    return sum(pytorchsim.cut_offset(pytorchsim.cut_digits(c), i) if c else i * s
               for i, s, c in zip(coord, dstride, cut))


def _collapse_keeps_every_address(shape, dstride, cut, want):
    """Collapse all axes into one; its cut must be `want` and address each coordinate as before."""
    import itertools
    import math
    module, op = _transfer(shape, dstride, cut)
    got = _entries(pytorchsim.renumbered_per_axis(op, [list(range(len(shape)))], shape))
    assert got == [want], f"collapsed cut {got}, want [{want}]"
    for n, coord in enumerate(itertools.product(*map(range, shape))):
        new = pytorchsim.cut_offset(pytorchsim.cut_digits(want), n)
        assert new == _address(shape, dstride, cut, coord), (coord, new)
    assert n + 1 == math.prod(shape)


def verifier_refuses_a_cut_of_the_wrong_rank():
    _refused([4, 8], [64, 1], [[0, 64, 192, 64]], "entries for a rank-2 tile")


def verifier_refuses_a_partial_digit_record():
    _refused([4, 8], [64, 1], [[0, 64, 192], []], "is not whole")


def verifier_refuses_an_outermost_radix_other_than_0():
    _refused([4, 8], [0, 1], [[4, 64, 0, 0, 2, 0, 0, 0], []], "outermost radix 0")


def verifier_refuses_an_inner_radix_of_1():
    _refused([4, 8], [0, 1], [[0, 64, 0, 0, 1, 0, 0, 0], []], "outermost radix 0")


def verifier_refuses_a_phase_on_a_digit_that_does_not_wrap():
    _refused([4, 8], [64, 1], [[0, 64, 0, 5], []], "outermost radix 0")


def verifier_refuses_an_inner_stride_off_the_dram_stride():
    _refused([4, 8], [64, 1], [[0, 64, 0, 0, 2, 32, 0, 0], []], "does not step the axis's dram_stride")


def collapse_joins_a_wrap_outside_an_uncut_mover():
    _collapse_keeps_every_address([4, 8], [64, 1], [[0, 64, 192, 64], []],
                                  [0, 64, 192, 64, 8, 1, 0, 0])


def collapse_joins_an_uncut_mover_outside_a_step():
    _collapse_keeps_every_address([2, 8], [1000, 0], [[], [0, 8, 0, 0, 4, 0, 0, 0]],
                                  [0, 1000, 0, 0, 2, 8, 0, 0, 4, 0, 0, 0])


def collapse_refuses_inner_digits_that_do_not_divide_the_extent():
    module, op = _transfer([2, 6], [1000, 0], [[], [0, 8, 0, 0, 4, 0, 0, 0]])
    try:
        pytorchsim.renumbered_per_axis(op, [[0, 1]], [2, 6])
    except NotImplementedError as e:
        assert "do not divide it" in str(e), e
        return
    raise AssertionError("a step of 4 was folded into an axis of extent 6")


def collapse_refuses_a_unit_axis_offset_folded_into_a_mover():
    module, op = _transfer([1, 8], [32, 1], [[0, 32, 384, 400], []])
    try:
        pytorchsim.renumbered_per_axis(op, [[0, 1]], [1, 8])
    except NotImplementedError as e:
        assert "unit axis" in str(e), e
        return
    raise AssertionError("a unit axis offsetting its coordinate by -384 was folded away")


def permute_keeps_a_unit_axis_offset_alone_in_its_group():
    module, op = _transfer([1, 8], [32, 1], [[0, 32, 384, 400], []])
    got = _entries(pytorchsim.renumbered_per_axis(op, [[1], [0]], [1, 8]))
    assert got == [[], [0, 32, 384, 400]], got


CASES = [verifier_refuses_a_cut_of_the_wrong_rank, verifier_refuses_a_partial_digit_record,
         verifier_refuses_an_outermost_radix_other_than_0, verifier_refuses_an_inner_radix_of_1,
         verifier_refuses_a_phase_on_a_digit_that_does_not_wrap,
         verifier_refuses_an_inner_stride_off_the_dram_stride,
         collapse_joins_a_wrap_outside_an_uncut_mover,
         collapse_joins_an_uncut_mover_outside_a_step,
         collapse_refuses_inner_digits_that_do_not_divide_the_extent,
         collapse_refuses_a_unit_axis_offset_folded_into_a_mover,
         permute_keeps_a_unit_axis_offset_alone_in_its_group]


def main():
    from mlir.ir import Context, Location
    bad = 0
    for case in CASES:
        ctx = Context()
        ctx.allow_unregistered_dialects = True
        with ctx, Location.unknown():
            try:
                case()
                print(f"ok    {case.__name__}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
