"""lib_pack: torchsim.pack, a lane's bank read as packed words -- `pack` of its elements to a 32-bit word,
the earlier element in the low bits; a register read through it is words from the start. The matrix unit
takes its operands so (ext/pytorchsim_tpu_family/lower_matmul)."""
from .lib_ir import elem_bits, name_of, strides_and_offset, walk

PACK_OP = "torchsim.pack"

WORD_BITS = 32


def pack(bank, why):
    """`bank`, a `[1, depth]` bank view, as `[1, depth / pack]` words; `why` names who asks, for
    the refusal of a bank that is not whole words."""
    from mlir.ir import IntegerType, MemRefType, Operation, StridedLayoutAttr
    mt = MemRefType(bank.type)
    each = WORD_BITS // elem_bits(mt.element_type)
    strides, offset = strides_and_offset(mt)
    depth = mt.shape[-1]
    if list(mt.shape[:-1]) != [1] or strides[-1] != 1 or offset % each or depth % each:
        raise NotImplementedError(
            f"torchsim.pack: {mt} from element {offset} is not one run of whole words of {each} "
            f"elements: {why}")
    words = depth // each
    ty = MemRefType.get([1, words], IntegerType.get_signless(WORD_BITS),
                        StridedLayoutAttr.get(offset // each, [words, 1]), mt.memory_space)
    return Operation.create(PACK_OP, results=[ty], operands=[bank]).results[0]


def lower_packs(module):
    """Every torchsim.pack written back as a change of type alone: lowered, a memref of either
    element is one descriptor, and the result's offset and strides are its type's."""
    from mlir.ir import InsertionPoint, Operation
    made = 0
    for op in list(walk(module.operation)):
        if name_of(op) != PACK_OP:
            continue
        res = op.results[0]
        with InsertionPoint(op):
            got = Operation.create("builtin.unrealized_conversion_cast", results=[res.type],
                                   operands=[op.operands[0]]).results[0]
        res.replace_all_uses_with(got)
        op.erase()
        made += 1
    return made
