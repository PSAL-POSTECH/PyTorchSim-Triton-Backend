"""torchsim-lower-to-llvm: the bank views and strided accesses lowered, the registered MLIR ->
LLVM-dialect passes run in process through the bindings PassManager, then the scratchpad reserved.
Runs after the lower-dma slot's pass; lower.TO_LLVM is the order."""

import os

SPAD_SECTION = ".spad"
SPAD_TOP_SECTION = ".spadtop"
SPAD_TOP_SYMBOL = "spad_bank_top"

SPAD_POISON_ENV = "TORCHSIM_COMPILE_SPAD_POISON"
SPAD_POISON_BYTES_SYMBOL = "torchsim_spad_poison_bytes"


class SpadReservationRefused(Exception):
    """No storage can be reserved for a global: a compile error, never a bank slice of the wrong
    size."""


def _refuse_a_scratchpad_life_left(module):
    from .lib_bank_layout import scratchpad_lives_left
    left = scratchpad_lives_left(module)
    if left:
        raise SpadReservationRefused(
            f"a scratchpad buffer is still allocated or freed by an op at LLVM lowering "
            f"({left[0].name}); finalize-memref-to-llvm would make it a malloc or a free on the "
            f"heap, not a slot in {SPAD_SECTION}. tritonpytorchsim-materialize-spad-allocs turns "
            f"each into a global first: {left[0]}")


STANDARD_STEPS = [
    "convert-linalg-to-loops",
    "convert-vector-to-scf{full-unroll=true}",
    "lower-affine",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "func.func(lower-vector-multi-reduction)",
    "convert-vector-to-llvm",
    "convert-arith-to-llvm",
    "convert-math-to-llvm",
    "convert-scf-to-cf",
    "convert-cf-to-llvm",
    "convert-func-to-llvm",
    "convert-arith-to-llvm",
    "convert-index-to-llvm",
    "reconcile-unrealized-casts",
]

STANDARD_PIPELINE = "builtin.module(" + ",".join(STANDARD_STEPS) + ")"

def _flatten(ty):
    """(element count, element type, lanes) of a nested `!llvm.array`: its dims are the memref's, so
    its lanes are lanes_of_shape's."""
    from mlir.dialects.llvm import ArrayType
    from .lib_bank_layout import lanes_of_shape
    shape = []
    while isinstance(ty, ArrayType):
        shape.append(ty.num_elements)
        ty = ty.element_type
    numel = 1
    for d in shape:
        numel *= d
    return numel, ty, lanes_of_shape(shape)


def _zero_tile(count, ety):
    from mlir.ir import (DenseElementsAttr, FloatAttr, FloatType, IntegerAttr,
                         IntegerType, RankedTensorType)
    if isinstance(ety, IntegerType):
        return DenseElementsAttr.get_splat(RankedTensorType.get([count], ety),
                                           IntegerAttr.get(ety, 0))
    if isinstance(ety, FloatType):
        return DenseElementsAttr.get_splat(RankedTensorType.get([count], ety),
                                           FloatAttr.get(ety, 0.0))
    raise SpadReservationRefused(
        f"a scratchpad global of `{ety}` has no zero this pass can write, so it "
        f"would stay an `external` declaration and the link would fail with an "
        f"undefined reference to it. Every tile this compiler banks is an integer or "
        f"a float -- a new element type needs its zero added to _zero_tile.")


def spad_poison():
    """The byte every .spad slot starts as, SPAD_POISON_ENV's; 0, the zero _zero_tile writes, when
    it is unset -- a non-zero one shows a kernel reading what nothing wrote."""
    said = os.environ.get(SPAD_POISON_ENV, "").strip()
    try:
        got = int(said, 0) if said else 0
    except ValueError:
        got = -1
    if not 0 <= got <= 0xFF:
        raise SpadReservationRefused(f"{SPAD_POISON_ENV}={said!r} is not a byte (0 is off)")
    return got


def _poison_tile(count, ety, byte):
    """`count` elements of `ety`, each `byte` in every one of its bytes."""
    import numpy as np
    from mlir.ir import DenseElementsAttr, FloatType, IntegerAttr, IntegerType, RankedTensorType
    width = ety.width
    bits = int.from_bytes(bytes([byte]) * -(-width // 8), "little") & ((1 << width) - 1)
    if isinstance(ety, IntegerType):
        signed = bits - (1 << width) if bits >> (width - 1) else bits
        return DenseElementsAttr.get_splat(RankedTensorType.get([count], ety),
                                           IntegerAttr.get(ety, signed))
    if isinstance(ety, FloatType) and width in (8, 16, 32, 64):
        word = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}[width]
        return DenseElementsAttr.get(np.full([1], bits, dtype=word), type=ety, shape=[count])
    return _zero_tile(count, ety)


def _mint_the_bank_top_anchor(top):
    from mlir.ir import (Attribute, DenseElementsAttr, InsertionPoint, IntegerAttr,
                         IntegerType, Operation, RankedTensorType, StringAttr, Type,
                         TypeAttr)
    i32 = IntegerType.get_signless(32)
    with InsertionPoint.at_block_begin(top):
        Operation.create(
            "llvm.mlir.global", results=[], operands=[], regions=1,
            attributes={
                "sym_name": StringAttr.get(SPAD_TOP_SYMBOL),
                "global_type": TypeAttr.get(Type.parse("!llvm.array<4 x i32>")),
                "linkage": Attribute.parse("#llvm.linkage<external>"),
                "addr_space": IntegerAttr.get(i32, 0),
                "section": StringAttr.get(SPAD_TOP_SECTION),
                "value": DenseElementsAttr.get_splat(
                    RankedTensorType.get([4], i32), IntegerAttr.get(i32, 0))})


def reserve_spad_storage(module, vectorlane, vlen_bits):
    """Every scratchpad global as a DEFINITION of ONE lane's share in `.spad`, starting on a
    vector-register boundary (a whole-register load never straddles two), plus the bank-top
    anchor."""
    from mlir.dialects.llvm import ArrayType
    from mlir.ir import IntegerAttr, IntegerType, StringAttr, TypeAttr
    from .lib_bank_layout import SPAD_SPACE, reserve_per_lane
    top = module.operation.regions[0].blocks[0]
    poison = spad_poison()
    reserved = []
    for op in [o.operation for o in top.operations
               if o.operation.name == "llvm.mlir.global"
               and IntegerAttr(o.operation.attributes["addr_space"]).value == SPAD_SPACE]:
        name = StringAttr(op.attributes["sym_name"]).value
        numel, elem, lanes = _flatten(TypeAttr(op.attributes["global_type"]).value)
        count = reserve_per_lane(numel, lanes, vectorlane)
        value = _poison_tile(count, elem, poison) if poison else _zero_tile(count, elem)
        op.attributes["global_type"] = TypeAttr.get(ArrayType.get(elem, count))
        op.attributes["value"] = value
        op.attributes["section"] = StringAttr.get(SPAD_SECTION)
        op.attributes["alignment"] = IntegerAttr.get(IntegerType.get_signless(64), vlen_bits // 8)
        reserved.append((name, count, elem))
    if reserved:
        _mint_the_bank_top_anchor(top)
    if reserved and poison:
        _mint_the_poisoned_extent(top, reserved, vlen_bits // 8)
    return reserved


def _mint_the_poisoned_extent(top, reserved, align):
    """SPAD_POISON_BYTES_SYMBOL, an i64 in .data: how many bytes from a bank's start the slots can
    reach, each aligned -- what the wrapper refills with the poison in every lane, every program."""
    from mlir.ir import (InsertionPoint, IntegerAttr, IntegerType, Operation, StringAttr,
                         TypeAttr, UnitAttr)
    from .lib_ir import elem_bytes
    i64 = IntegerType.get_signless(64)
    reach = sum(-(-count * elem_bytes(elem) // align) * align for _, count, elem in reserved)
    anchor = next(o.operation for o in top.operations if o.operation.name == "llvm.mlir.global")
    with InsertionPoint.at_block_begin(top):
        Operation.create(
            "llvm.mlir.global", results=[], operands=[], regions=1,
            attributes={"sym_name": StringAttr.get(SPAD_POISON_BYTES_SYMBOL),
                        "global_type": TypeAttr.get(i64),
                        "linkage": anchor.attributes["linkage"],
                        "constant": UnitAttr.get(),
                        "value": IntegerAttr.get(i64, reach)})


def lower_strided_accesses(module):
    """Every torchsim.strided_load/store -> llvm.intr.experimental.vp.strided.load/store at the
    first element's byte address, `stride` elements apart, reading no more than `count`; the
    pointer is in SPAD_SPACE when the memref is, so a scratchpad access is a space-1 one."""
    from mlir.dialects import arith, llvm, memref
    from mlir.ir import (DenseElementsAttr, IndexType, InsertionPoint, IntegerAttr, IntegerType,
                         Operation, ShapedType, VectorType)
    from .lib_bank_layout import SPAD_SPACE, in_scratchpad
    from .lib_ir import elem_bytes, strides_and_offset, walk
    idx, i1, i32, i64 = (IndexType.get(), IntegerType.get_signless(1),
                         IntegerType.get_signless(32), IntegerType.get_signless(64))
    for op in [o for o in walk(module.operation)
               if o.name in ("torchsim.strided_load", "torchsim.strided_store")]:
        load = op.name == "torchsim.strided_load"
        vec = op.results[0] if load else op.operands[0]
        mem = op.operands[0 if load else 1]
        indices = list(op.operands)[1 if load else 2:]
        n = VectorType(vec.type).shape[0]
        esize = elem_bytes(VectorType(vec.type).element_type)
        so = strides_and_offset(mem)
        if so is None or ShapedType.get_dynamic_stride_or_offset() in list(so[0]) + [so[1]]:
            raise NotImplementedError(f"a strided access through a memref with no static layout: {op}")
        with InsertionPoint(op):
            def cst(ty, v):
                return arith.ConstantOp(ty, IntegerAttr.get(ty, v)).result
            off = cst(idx, so[1])
            for i, st in zip(indices, so[0]):
                off = arith.AddIOp(off, arith.MulIOp(i, cst(idx, st)).result).result
            addr = arith.AddIOp(memref.ExtractAlignedPointerAsIndexOp(mem).result,
                                arith.MulIOp(off, cst(idx, esize)).result).result
            space = SPAD_SPACE if in_scratchpad(mem.type) else 0
            ptr = llvm.IntToPtrOp(llvm.PointerType.get(space),
                                  arith.IndexCastOp(i64, addr).result).result
            mask = arith.ConstantOp(VectorType.get([n], i1), DenseElementsAttr.get_splat(
                VectorType.get([n], i1), IntegerAttr.get(i1, 1))).result
            tail = [cst(i64, IntegerAttr(op.attributes["stride"]).value * esize), mask,
                    cst(i32, min(IntegerAttr(op.attributes["count"]).value, n))]
            if load:
                got = Operation.create("llvm.intr.experimental.vp.strided.load",
                                       results=[vec.type], operands=[ptr] + tail).results[0]
                op.results[0].replace_all_uses_with(got)
            else:
                Operation.create("llvm.intr.experimental.vp.strided.store", results=[],
                                 operands=[vec, ptr] + tail)
        op.operation.erase()


def run(module, *, vectorlane, target, dump=None, **_):
    """Lower `module` to the LLVM dialect in place and return the scratchpad reservations;
    `dump(name, module)`, when given, runs after each standard step, one at a time."""
    from mlir.passmanager import PassManager
    from .lib_bank_layout import lower_bank_views
    from .lib_pack import lower_packs
    ctx = module.context
    _refuse_a_scratchpad_life_left(module)
    lower_bank_views(module)
    lower_packs(module)
    lower_strided_accesses(module)
    if dump is None:
        PassManager.parse(STANDARD_PIPELINE, ctx).run(module.operation)
    else:
        for step in STANDARD_STEPS:
            PassManager.parse(f"builtin.module({step})", ctx).run(module.operation)
            dump(step.split("{")[0].split("(")[-1].rstrip(")"), module)
    reserved = reserve_spad_storage(module, vectorlane, target.vlen_bits)
    if dump is not None:
        dump("torchsim-reserve-spad-storage", module)
    return reserved
