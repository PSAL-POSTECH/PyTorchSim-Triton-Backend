"""to_vcix: the TPU's unit ops (pytorchsim_tpu) as the VCIX instructions they are, an encoder an op in
TO_VCIX, which convert-to-vcix applies; opcodes and immediates are the Coprocessor model's."""

# The MXU's immediate bits beside the operand format (examples/tpu/mxu.hpp).
SECOND = 4      # a vmatprep of an input, not of weights; a vpop of the upper columns
LATCH = 4       # a vmatmul that first makes the staged weights the active
FIRST_HALF, SECOND_HALF = 0x08, 0x10    # which half of the depth a vmatpush stages, at [4:3]

# Each XLU move's push and pop by its (pre, XU, post) setting, the SIMM5 (pre << 3) | (XU << 2) | post, a
# reduce's pop by 0; each reduce kind by the field xlu_push_reduce takes.
RPU_BYPASS, RPU_REPLICATE, RPU_ARBITRARY = 0, 1, 2
XLU_FIELDS = {"permute": (RPU_ARBITRARY, 0, RPU_BYPASS), "gather": (RPU_BYPASS, 1, RPU_REPLICATE),
              "transpose": (RPU_BYPASS, 1, RPU_BYPASS)}
XLU_REDUCE_FIELD = {"max": 0, "sum": 1, "min": 2}

# Each SFU function's sf.vc.v.iv (opcode, imm) (examples/tpu/sfu.hpp).
SFU_ENCODINGS = {"exp": (3, 0), "erf": (0, 0), "tanh": (1, 0), "sin": (2, 0), "cos": (2, 1), "log": (2, 2),
                 "atan": (2, 3)}


def _int_attr(bits, value):
    import mlir.ir as ir
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(bits), value)


def _int(attr):
    import mlir.ir as ir
    return ir.IntegerAttr(attr).value


def _bool(attr):
    import mlir.ir as ir
    return ir.BoolAttr(attr).value


def _str(attr):
    import mlir.ir as ir
    return ir.StringAttr(attr).value


def _simm5(field):
    """The field at 19:15 as the signed 5-bit immediate the op carries: bit 4 set reads negative."""
    return field - 32 if field & 0x10 else field


def _xlu_mode_imm(mode):
    if mode == "reduce":
        return 0
    pre, xu, post = XLU_FIELDS[mode]
    return _simm5((pre << 3) | (xu << 2) | post)


def _vl_of(ty):
    """A vector type's length, as the i64 an EUP push or pop takes for vl."""
    import mlir.ir as ir
    return _int_attr(64, ir.VectorType(ty).shape[0])


def _length(ty):
    """The length of a 1-D vector type; None for any other."""
    import mlir.ir as ir
    return ir.VectorType(ty).shape[0] if isinstance(ty, ir.VectorType) and ir.VectorType(ty).rank == 1 else None


def _carried(values):
    """`values` as a vcix op takes them: a vector in the unit's vector (vcix.to_unit), its vl operand
    saying how much of it counts -- llama4's RMSNorm folds 40 words a lane across the lanes."""
    from ...passes.utils.vcix import to_unit
    return [to_unit(v, _length(v.type)) if _length(v.type) else v for v in values]


def _unit_types_of(types):
    """`types` with each vector type the unit's vector of its length (vcix.unit_vector)."""
    import mlir.ir as ir
    from ...passes.utils.vcix import unit_vector
    return [ir.VectorType.get([unit_vector(_length(t))], ir.VectorType(t).element_type) if _length(t) else t
            for t in types]


def _unit_types(results):
    return _unit_types_of([r.type for r in results])


def _replace_results(op, new):
    """`op`'s results taken from `new`'s, each cut back from the unit's vector (vcix.from_unit)."""
    from ...passes.utils.vcix import from_unit
    for old, got in zip(op.results, new.results):
        old.replace_all_uses_with(from_unit(got, _length(old.type)) if _length(old.type) else got)
    op.erase()


def _replace(op, name, attrs, operands=None):
    """`op` replaced by a `name` op of its operands (or `operands`) and result types, with `attrs`."""
    import mlir.ir as ir
    new = ir.Operation.create(name, results=_unit_types(op.results),
                              operands=_carried(list(op.operands) if operands is None else operands),
                              attributes=attrs)
    _replace_results(op, new)


def _const(attr, ty=None):
    import mlir.ir as ir
    return ir.Operation.create("arith.constant", results=[ty or ir.IntegerType.get_signless(64)],
                               attributes={"value": attr}).result


def _intrinsic(op, intrin, operands, results=()):
    """`op` replaced by llvm.call_intrinsic `intrin` of `operands`, a form the vcix dialect does not name."""
    import mlir.ir as ir
    operands = _carried(operands)
    new = ir.Operation.create("llvm.call_intrinsic", results=_unit_types_of(results), operands=operands,
                              attributes={
        "intrin": ir.StringAttr.get(intrin), "op_bundle_sizes": ir.DenseI32ArrayAttr.get([]),
        "operandSegmentSizes": ir.DenseI32ArrayAttr.get([len(operands), 0])})
    _replace_results(op, new)


def _lane_idx(op):
    _replace(op, "vcix.v.i", {"opcode": _int_attr(64, 0), "rs2": _int_attr(64, 0), "imm": _int_attr(64, 0)})


def _pack(op):
    """MISC: the bf16 pack and its unpack (examples/tpu/misc.hpp), sf.vc.v.vv / sf.vc.v.iv opcode 2."""
    _replace(op, "vcix.v.sv", {"opcode": _int_attr(64, 2)})


def _unpack(imm):
    return lambda op: _replace(op, "vcix.v.iv", {"opcode": _int_attr(64, 2), "imm": _int_attr(32, imm)})


def _sfu(opcode, imm):
    return lambda op: _replace(op, "vcix.v.iv", {"opcode": _int_attr(64, opcode), "imm": _int_attr(32, imm)})


def _eup_push(op):
    """EUP (the v6e's, examples/tpuv6e): sfu_push is sf.vc.vv opcode 3 reading vs2, vl the vector's length."""
    x = op.operands[0]
    opcode, none, vl = _const(_int_attr(64, 3)), _const(_int_attr(64, 0)), _const(_vl_of(x.type))
    _intrinsic(op, "llvm.riscv.sf.vc.vv.se", [opcode, none, x, x, vl])


def _eup_pop(op):
    """sfu_pop is sf.vc.v.x opcode 3 writing vd."""
    ty = op.results[0].type
    opcode, none, vl = _const(_int_attr(64, 3)), _const(_int_attr(64, 0)), _const(_vl_of(ty))
    _intrinsic(op, "llvm.riscv.sf.vc.v.x.se", [opcode, none, none, vl], [ty])


def _mxu(opcode, imm):
    """MXU (examples/tpu/mxu.hpp): vmatmul, vmatpush, vmatprep are sf.vc.iv opcodes 0, 1, 2; each immediate is
    the operand format with the instruction's own bits beside it."""
    return lambda op: _replace(op, "vcix.iv", {"opcode": _int_attr(64, opcode), "imm": _int_attr(64, imm(op)),
                                               "rd": _int_attr(64, 0)})


def _mxu_pop(op):
    """vpop is sf.vc.v.i opcode 2, the upper columns' with SECOND."""
    imm = SECOND if _bool(op.attributes["upper"]) else 0
    _replace(op, "vcix.v.i", {"opcode": _int_attr(64, 2), "imm": _int_attr(64, imm), "rs2": _int_attr(64, 0)})


def _xlu_push(op):
    """XLU (examples/tpu/xlu.hpp): xlu_push is sf.vc.iv opcode 3, carrying the pass's (pre, XU, post) bits."""
    imm = _xlu_mode_imm(_str(op.attributes["mode"]))
    _replace(op, "vcix.iv", {"opcode": _int_attr(64, 3), "imm": _int_attr(64, imm), "rd": _int_attr(64, 0)})


def _xlu_pop(op):
    """xlu_pop is sf.vc.v.i opcode 1, with the push's bits."""
    imm = _xlu_mode_imm(_str(op.attributes["mode"]))
    _replace(op, "vcix.v.i", {"opcode": _int_attr(64, 1), "imm": _int_attr(64, imm), "rs2": _int_attr(64, 0)})


def _xlu_set_pattern(op):
    """xlu_set_pattern is sf.vc.ivv opcode 3."""
    lanes, vl = op.operands
    opcode, none = _const(_int_attr(64, 3)), _const(_int_attr(64, 0))
    _intrinsic(op, "llvm.riscv.sf.vc.ivv.se", [opcode, lanes, lanes, none, vl])


def _xlu_push_reduce(op):
    """xlu_push_reduce is sf.vc.ivv opcode 2, the reduce kind its immediate."""
    import mlir.ir as ir
    w, vl = op.operands
    vt = ir.VectorType(w.type)
    opcode = _const(_int_attr(64, 2))
    zero = _const(ir.DenseElementsAttr.get_splat(vt, ir.IntegerAttr.get(vt.element_type, 0)), w.type)
    kind = _const(_int_attr(64, XLU_REDUCE_FIELD[_str(op.attributes["kind"])]))
    _intrinsic(op, "llvm.riscv.sf.vc.ivv.se", [opcode, zero, w, kind, vl])


def _prep_imm(op):
    return _int(op.attributes["format"]) | (SECOND if _bool(op.attributes["input"]) else 0)


def _push_imm(op):
    half = SECOND_HALF if _bool(op.attributes["second_half"]) else FIRST_HALF
    return _simm5(_int(op.attributes["format"]) | half)


def _matmul_imm(op):
    return _int(op.attributes["format"]) | (LATCH if _bool(op.attributes["latch"]) else 0)


#: unit op name -> its encoder, which replaces the op where it stands.
TO_VCIX = {
    "pytorchsim_tpu.misc.lane_idx": _lane_idx,
    "pytorchsim_tpu.misc.pack": _pack,
    "pytorchsim_tpu.misc.unpack_lo": _unpack(8),
    "pytorchsim_tpu.misc.unpack_hi": _unpack(9),
    **{f"pytorchsim_tpu.sfu.{fn}": _sfu(*code) for fn, code in SFU_ENCODINGS.items()},
    "pytorchsim_tpu.eup.push": _eup_push,
    "pytorchsim_tpu.eup.pop": _eup_pop,
    "pytorchsim_tpu.mxu.prep": _mxu(2, _prep_imm),
    "pytorchsim_tpu.mxu.push": _mxu(1, _push_imm),
    "pytorchsim_tpu.mxu.matmul": _mxu(0, _matmul_imm),
    "pytorchsim_tpu.mxu.pop": _mxu_pop,
    "pytorchsim_tpu.xlu.push": _xlu_push,
    "pytorchsim_tpu.xlu.pop": _xlu_pop,
    "pytorchsim_tpu.xlu.set_pattern": _xlu_set_pattern,
    "pytorchsim_tpu.xlu.push_reduce": _xlu_push_reduce,
}
