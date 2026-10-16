"""torchsim-lower-to-llvm: the bank views and strided accesses lowered, the registered MLIR ->
LLVM-dialect passes run in process through the bindings PassManager, then the scratchpad reserved.
Runs after the lower-dma slot's pass; lower.TO_LLVM is the order."""
import re

SPAD_SECTION = ".spad"
SPAD_TOP_SECTION = ".spadtop"
SPAD_TOP_SYMBOL = "spad_bank_top"

_LANES_TAG_RE = re.compile(r"_(\d+)lane$")


class SpadReservationRefused(Exception):
    """No storage can be reserved for a global: a compile error, never a bank slice of the wrong
    size."""


def _refuse_if_the_name_disagrees(name, lanes):
    tag = _LANES_TAG_RE.search(name)
    if tag is not None and int(tag.group(1)) != lanes:
        raise SpadReservationRefused(
            f"{name} says {tag.group(1)} lane(s) but its type leads with {lanes}: "
            f"a scratchpad global's leading extent IS its lane spread, so one of "
            f"the two is stale. A pass that grows such a buffer must add an axis, "
            f"not overwrite the leading one -- reserving {lanes} lanes' worth for "
            f"a tile that sits in {tag.group(1)} would come out short and silent.")


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
    """(element count, element type, lanes) of a nested `!llvm.array`: its outermost dim is the lanes."""
    from mlir.dialects.llvm import ArrayType
    numel, lanes = 1, None
    while isinstance(ty, ArrayType):
        lanes = ty.num_elements if lanes is None else lanes
        numel *= ty.num_elements
        ty = ty.element_type
    return numel, ty, lanes or 1


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
    reserved = []
    for op in [o.operation for o in top.operations
               if o.operation.name == "llvm.mlir.global"
               and IntegerAttr(o.operation.attributes["addr_space"]).value == SPAD_SPACE]:
        name = StringAttr(op.attributes["sym_name"]).value
        numel, elem, lanes = _flatten(TypeAttr(op.attributes["global_type"]).value)
        _refuse_if_the_name_disagrees(name, lanes)
        count = reserve_per_lane(numel, lanes, vectorlane)
        value = _zero_tile(count, elem)
        op.attributes["global_type"] = TypeAttr.get(ArrayType.get(elem, count))
        op.attributes["value"] = value
        op.attributes["section"] = StringAttr.get(SPAD_SECTION)
        op.attributes["alignment"] = IntegerAttr.get(IntegerType.get_signless(64), vlen_bits // 8)
        reserved.append((name, count, elem))
    if reserved:
        _mint_the_bank_top_anchor(top)
    return reserved


def lower_strided_accesses(module):
    """Every torchsim.strided_load/store -> llvm.intr.experimental.vp.strided.load/store at the
    first element's byte address, `stride` elements apart, reading no more than `count`."""
    from mlir.dialects import arith, llvm, memref
    from mlir.ir import (DenseElementsAttr, IndexType, InsertionPoint, IntegerAttr, IntegerType,
                         Operation, ShapedType, VectorType)
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
            ptr = llvm.IntToPtrOp(llvm.PointerType.get(), arith.IndexCastOp(i64, addr).result).result
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
