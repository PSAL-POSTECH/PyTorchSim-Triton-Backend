"""pack: pytorchsim_tpu.vpu.packed_view, a lane's bank read as packed words -- `pack` of its
elements to a 32-bit word, the earlier element in the low bits; a register read through it is words
from the start. The matrix unit takes its operands so (targets/tpu/lower_matmul)."""
from .pytorchsim_tpu import PACKED_VIEW
from ...passes.utils.ir_utils import elem_bits, strides_and_offset

PACK_OP = PACKED_VIEW

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
            f"{PACK_OP}: {mt} from element {offset} is not one run of whole words of {each} "
            f"elements: {why}")
    words = depth // each
    ty = MemRefType.get([1, words], IntegerType.get_signless(WORD_BITS),
                        StridedLayoutAttr.get(offset // each, [words, 1]), mt.memory_space)
    return Operation.create(PACK_OP, results=[ty], operands=[bank]).results[0]

