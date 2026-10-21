"""tritonpytorchsim-keep-spad-accesses-vector-tpuv6e: address space 1 is banked per lane, so every
vector or scalar store and vector load through a !llvm.ptr<1> becomes llvm.vp.*, which every lane's
bank takes and llc never scalarizes to a lane-0 lw/sw; what cannot is refused, by name."""

from ...passes.lib_bank_layout import SPAD_SPACE, SpadAccessRefused
from ...passes.lib_ir import walk
from ...passes.lower import KEEP_SPAD_ACCESSES_VECTOR

NAME = "tritonpytorchsim-keep-spad-accesses-vector-tpuv6e"

REQUIRES = ("vcix_tpuv6e",)

SLOT = KEEP_SPAD_ACCESSES_VECTOR.name

FORBIDS = ()

REWRITE = "rewrite"

INDEXED = ("an indexed (gather/scatter) access on the scratchpad: each lane's address is its own, "
           "which a lane's bank does not see")
MEMORY_INTRINSIC = ("a memory intrinsic on the scratchpad, which llc expands into scalar accesses "
                    "that reach lane 0's bank only")
SCALABLE = "a scalable vector on the scratchpad, whose element count is no constant EVL"
SUB_BYTE = ("a vector of sub-byte elements on the scratchpad: llc crashes on vp of them, and a "
            "memref of them holds a byte per element where the vector packs bits")
ALL_TRUE_MASK = ("a masked access on the scratchpad whose mask is a constant all-true, which llc "
                 "folds to a plain load/store it may scalarize")
VOLATILE = "a volatile access on the scratchpad, which vp cannot carry"
NONTEMPORAL = "a nontemporal access on the scratchpad, which vp cannot carry"
NOT_A_NUMBER = "a store of a scalar that is no integer or float on the scratchpad, which no vp carries"
ZERO_STRIDE = ("a strided access on the scratchpad whose stride is a constant 0, which llc may fold "
               "into a scalar access and a splat that reach lane 0's bank only")
SURVIVED = "a vector access or a scalar store on the scratchpad survives the rewrite"


def run(module, **_):
    """Rewrite every vector load/store and scalar store on space 1 to vp, after refusing the module if
    any space-1 access falls in a refused class; then check no such access is left."""
    found = [(op, _verdict(op)) for op in walk(module.operation)]
    for op, why in found:
        if why not in (None, REWRITE):
            raise SpadAccessRefused(f"{NAME}: {why}, at {op.location}: {_short(op)}", reason=why)
    for op, why in found:
        if why == REWRITE:
            _to_vp(op)
    left = [op for op in walk(module.operation) if _verdict(op) is not None]
    if left:
        raise SpadAccessRefused(f"{NAME}: {SURVIVED} ({len(left)}), first at {left[0].location}: "
                                f"{_short(left[0])}", reason=SURVIVED)
    return module


def _verdict(op):
    """None for an op left as it is, REWRITE for a space-1 vector access or scalar store to turn into
    vp, else the reason it is refused, read from types; a scalar integer is widened to whole bytes."""
    from mlir.dialects import llvm
    from mlir.ir import IntegerType, VectorType
    if any(_spad_vector_of_pointers(v.type) for v in op.operands):
        return INDEXED
    view = op.opview
    if isinstance(view, (llvm.VPStridedLoadOp, llvm.VPStridedStoreOp)):
        return ZERO_STRIDE if _is_spad(view.ptr.type) and _constant_zero(view.stride) else None
    if isinstance(view, (llvm.MemcpyOp, llvm.MemmoveOp, llvm.MemsetOp, llvm.MemcpyInlineOp,
                         llvm.MemsetInlineOp)):
        return MEMORY_INTRINSIC if any(_is_spad(v.type) for v in op.operands) else None
    if isinstance(view, (llvm.LoadOp, llvm.StoreOp)):
        addr, value = view.addr, (view.res if isinstance(view, llvm.LoadOp) else view.value)
    elif isinstance(view, (llvm.MaskedLoadOp, llvm.MaskedStoreOp)):
        addr, value = view.data, (view.res if isinstance(view, llvm.MaskedLoadOp) else view.value)
    else:
        return None
    if not _is_spad(addr.type):
        return None
    if not isinstance(value.type, VectorType):
        if not isinstance(view, llvm.StoreOp):
            return None
        if not _is_number(value.type):
            return NOT_A_NUMBER
        elem, widened = value.type, isinstance(value.type, IntegerType)
    else:
        vty = VectorType(value.type)
        if vty.scalable:
            return SCALABLE
        elem, widened = vty.element_type, False
    if _elem_bits(elem) % 8 and not widened:
        return SUB_BYTE
    if isinstance(view, (llvm.MaskedLoadOp, llvm.MaskedStoreOp)):
        return ALL_TRUE_MASK if _all_true(view.mask) else REWRITE
    if "volatile_" in op.attributes:
        return VOLATILE
    if "nontemporal" in op.attributes:
        return NONTEMPORAL
    return REWRITE


def _to_vp(op):
    """`op`, a vector load/store or a scalar store on space 1, as llvm.vp.load/vp.store under its
    location: a scalar as element 0 of a vector<1> (an integer zero-extended to its store size in whole
    bytes), the mask its own or all-true, EVL the element count, a masked load's off lanes its passthru."""
    from mlir.dialects import llvm
    from mlir.ir import (ArrayAttr, DenseElementsAttr, DictAttr, InsertionPoint, IntegerAttr,
                         IntegerType, VectorType)
    view = op.opview
    load = isinstance(view, (llvm.LoadOp, llvm.MaskedLoadOp))
    masked = isinstance(view, (llvm.MaskedLoadOp, llvm.MaskedStoreOp))
    addr = view.data if masked else view.addr
    value = view.res if load else view.value
    i1, i32, i64 = (IntegerType.get_signless(w) for w in (1, 32, 64))
    align = op.attributes["alignment"] if "alignment" in op.attributes else None
    ptr_attr = DictAttr.get({"llvm.align": IntegerAttr.get(i64, IntegerAttr(align).value)}
                            if align is not None else {})
    with InsertionPoint(op), op.location:
        if not isinstance(value.type, VectorType):
            if isinstance(value.type, IntegerType) and IntegerType(value.type).width % 8:
                value = llvm.ZExtOp(IntegerType.get_signless(-(-IntegerType(value.type).width // 8) * 8),
                                    value).result
            zero = llvm.ConstantOp(i32, IntegerAttr.get(i32, 0)).result
            value = llvm.InsertElementOp(llvm.UndefOp(VectorType.get([1], value.type)).result,
                                         value, zero).result
        vty = VectorType(value.type)
        if masked:
            mask = view.mask
        else:
            mty = VectorType.get(vty.shape, i1)
            mask = llvm.ConstantOp(mty, DenseElementsAttr.get_splat(mty, IntegerAttr.get(i1, 1))).result
        evl = llvm.ConstantOp(i32, IntegerAttr.get(i32, vty.shape[0])).result
        if load:
            got = llvm.CallIntrinsicOp(vty, "llvm.vp.load", [addr, mask, evl], [], [],
                                       arg_attrs=ArrayAttr.get([ptr_attr, DictAttr.get(),
                                                                DictAttr.get()])).results[0]
            if masked and view.pass_thru is not None:
                got = llvm.SelectOp(mask, got, view.pass_thru).result
            op.results[0].replace_all_uses_with(got)
        else:
            llvm.CallIntrinsicOp(None, "llvm.vp.store", [value, addr, mask, evl], [], [],
                                 arg_attrs=ArrayAttr.get([DictAttr.get(), ptr_attr, DictAttr.get(),
                                                          DictAttr.get()]))
    op.erase()


def _is_spad(ty):
    from mlir.dialects import llvm
    return isinstance(ty, llvm.PointerType) and llvm.PointerType(ty).address_space == SPAD_SPACE


def _spad_vector_of_pointers(ty):
    from mlir.ir import VectorType
    return isinstance(ty, VectorType) and _is_spad(VectorType(ty).element_type)


def _is_number(ty):
    from mlir.ir import FloatType, IntegerType
    return isinstance(ty, (IntegerType, FloatType))


def _constant_zero(v):
    """Whether `v` is an llvm.mlir.constant or arith.constant integer 0."""
    from mlir.dialects import arith, llvm
    from mlir.ir import IntegerAttr, OpResult
    if not isinstance(v, OpResult) or not isinstance(v.owner.opview, (llvm.ConstantOp, arith.ConstantOp)):
        return False
    value = v.owner.attributes["value"]
    return isinstance(value, IntegerAttr) and IntegerAttr(value).value == 0


def _elem_bits(ty):
    """The width of a vector element in bits; 1 (refused) for one with no width, such as a pointer."""
    from mlir.ir import FloatType, IntegerType
    for kind in (IntegerType, FloatType):
        if isinstance(ty, kind):
            return kind(ty).width
    return 1


def _all_true(mask):
    """Whether `mask` is an llvm.mlir.constant splat of true."""
    from mlir.dialects import llvm
    from mlir.ir import DenseElementsAttr, IntegerAttr, OpResult
    if not isinstance(mask, OpResult) or not isinstance(mask.owner.opview, llvm.ConstantOp):
        return False
    value = mask.owner.attributes["value"]
    if not isinstance(value, DenseElementsAttr) or not DenseElementsAttr(value).is_splat:
        return False
    return IntegerAttr(DenseElementsAttr(value).get_splat_value()).value != 0


def _short(op):
    return " ".join(str(op).split())[:200]
