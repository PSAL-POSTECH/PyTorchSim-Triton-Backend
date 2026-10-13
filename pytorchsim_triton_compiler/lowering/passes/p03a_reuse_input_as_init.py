"""tritonpytorchsim-reuse-input-as-init: an all-parallel linalg op whose body never reads its init
takes as init an input of the same type read through the same map, so bufferize may write where it
reads. Whether it does is one-shot-bufferize's conflict analysis; no candidate, no change."""

from .lib_ir import (funcs, indexing_maps, iterator_types, op_of, operand_segments,
                     parallel_iterator, unread_destinations, walk)

NAME = "tritonpytorchsim-reuse-input-as-init"


def run(module, **_opts):
    """Every op that states its maps and iterators is read; none is recognised by name."""
    for func in funcs(module):
        for op in walk(func.operation):
            _rewrite_unread_inits_to_inputs(op)
    return module


def _rewrite_unread_inits_to_inputs(op):
    maps, iters = indexing_maps(op), iterator_types(op)
    if maps is None or iters is None or "operandSegmentSizes" not in op.attributes:
        return
    if any(t != str(parallel_iterator()) for t in iters):
        return
    n_in = operand_segments(op)[0]
    if len(maps) != len(op.operands) or len(op.results) != len(op.operands) - n_in:
        return
    for slot in sorted(unread_destinations(op)):
        found = _get_candidates(op, slot, n_in, maps)
        if found:
            op.operands[slot] = found[0]


def _get_candidates(op, slot, n_in, maps):
    """The inputs that can stand as the init at `slot`, best first: the one the block hands back,
    then one nothing else uses, then operand order. An input read after `op` is not one."""
    out = op.operands[slot]
    inits = [op.operands[k] for k in range(n_in, len(op.operands))]
    ranked = []
    for k in range(n_in):
        v = op.operands[k]
        if v.type != out.type or maps[k] != maps[slot] or v in inits:
            continue
        others = [op_of(u.owner) for u in v.uses if op_of(u.owner) != op]
        if not _is_last_read_here(v, op, others):
            continue
        ranked.append((not _is_handed_back(v, op.results[slot - n_in]), bool(others), k))
    return [op.operands[k] for _, _, k in sorted(ranked)]


def _is_last_read_here(v, op, others):
    """`v` is made in `op`'s own block and every other user sits before `op` in it: the value is
    dead once `op` has read it, on this trip and on the next."""
    from mlir.ir import BlockArgument, InsertionPoint
    block = InsertionPoint(op).block
    made_in = v.owner if isinstance(v, BlockArgument) else InsertionPoint(op_of(v.owner)).block
    if made_in != block:
        return False
    before = []
    for o in block.operations:
        if op_of(o) == op:
            break
        before.append(op_of(o))
    for user in others:
        while InsertionPoint(user).block != block:
            user = op_of(user.parent)
        if user not in before:
            return False
    return True


def _is_handed_back(v, result):
    """`v` is an argument of the block whose terminator takes `result`: a loop's carried value and
    the value the trip yields for it."""
    from mlir.ir import BlockArgument
    if not isinstance(v, BlockArgument):
        return False
    last = list(v.owner.operations)[-1].operation
    return any(o == result for o in last.operands)
