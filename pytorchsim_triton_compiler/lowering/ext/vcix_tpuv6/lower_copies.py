"""tritonpytorchsim-lower-copies-tpuv6: every `torchsim.copy` onto one move of vcix_tpuv6's
cross-lane unit (tpu::Xlu), stamped with the XLU op, the rounds that carry it and the kind of move
it is; a copy no XLU op carries is refused -- this model has no other carrier for it."""

import itertools
import math

from ....dialects import torchsim
from ...passes.lib_bank_layout import row_major
from ...passes.lib_ir import funcs, int_array, name_of, op_of, walk

NAME = "tritonpytorchsim-lower-copies-tpuv6"

REQUIRES = ("vcix_tpuv6",)

SLOT = "lower-copies"

FORBIDS = (torchsim.CopyOp.OPERATION_NAME,)


def run(module, lanes, **_opts):
    """Each copy, in the order it stands, onto its carrier."""
    for f in funcs(module):
        for op in [op_of(o) for o in walk(f.operation)
                   if name_of(o) == torchsim.CopyOp.OPERATION_NAME]:
            _rewrite_copy(op, lanes)
    return module


def _rewrite_copy(op, lanes):
    from .lib_xlu import XLU_PERMUTE, choose_xlu_op
    kind = _get_copy_kind(op)
    src, dst = op.operands[0], op.operands[1]
    s_shape, d_shape = list(src.type.shape), list(dst.type.shape)
    kinds = (XLU_PERMUTE,) if "into_window" in op.attributes else None
    chosen = choose_xlu_op(_get_pairs(op, kind), s_shape, d_shape, lanes, kinds=kinds)
    if chosen is None:
        raise NotImplementedError(f"a {kind} copy no XLU op carries: {op}")
    _build_move(op, chosen, src, dst)
    op.operation.erase()


def _get_copy_kind(op):
    if "added_dims" in op.attributes:
        return "broadcast"
    if "static_offsets" in op.attributes:
        return "window"
    src, dst = op.operands[0], op.operands[1]
    if (_get_logical(list(src.type.shape), _get_perm(op, "src_perm"))
            != _get_logical(list(dst.type.shape), _get_perm(op, "dst_perm"))):
        return "reshape"
    return "relayout"


def _get_pairs(op, kind):
    src, dst = op.operands[0], op.operands[1]
    ps, pd = _get_perm(op, "src_perm"), _get_perm(op, "dst_perm")
    ss, ds = _get_logical(list(src.type.shape), ps), _get_logical(list(dst.type.shape), pd)
    at_s, at_d = _get_positions(list(src.type.shape)), _get_positions(list(dst.type.shape))
    return [(at_s[_get_flat(ss, ps, a)], at_d[_get_flat(ds, pd, b)])
            for a, b in _get_elements(op, kind, ss, ds)]


def _get_elements(op, kind, ss, ds):
    from ...passes.lib_indexing_maps import get_kept_axes
    if kind == "broadcast":
        added = set(int_array(op.attributes["added_dims"]))
        kept = [d for d in range(len(ds)) if d not in added]
        return [([y[d] for d in kept], y) for y in _get_all_coords(ds)]
    if kind == "window":
        into = "into_window" in op.attributes
        offs, sizes, steps = _get_window(op)
        small, big = (ss, ds) if into else (ds, ss)
        kept = get_kept_axes(sizes, small)
        out = []
        for y in _get_all_coords(small):
            x = list(offs)
            for j, a in enumerate(kept):
                x[a] += steps[a] * y[j]
            out.append((y, x) if into else (x, y))
        return out
    if kind == "reshape":
        return [(_get_coords(f, ss), _get_coords(f, ds)) for f in range(math.prod(ss))]
    return [(y, y) for y in _get_all_coords(ds)]


def _get_positions(shape):
    if not shape:
        return [(0, 0)]
    rows = row_major(shape)
    return [(x[0], sum(x[i] * rows[i] for i in range(1, len(shape))))
            for x in _get_all_coords(shape)]


def _build_move(op, chosen, src, dst):
    """The move, stamped with its XLU op, its rounds and, for whoever counts moves without
    knowing this unit, the kind of move that op is."""
    from mlir.ir import (ArrayAttr, DenseI64ArrayAttr, InsertionPoint, IntegerAttr, IntegerType,
                         Operation, StringAttr)
    from ...passes.lib_cross_lane import (CROSS_LANE_ATTR, CROSS_LANE_KIND, CROSS_LANE_OP,
                                          CROSS_LANE_ROUNDS)
    from .lib_xlu import emitter_of
    xlu_op, plan = chosen
    with InsertionPoint(op):
        move = Operation.create(CROSS_LANE_OP, operands=[src, dst])
    move.attributes[CROSS_LANE_ATTR] = IntegerAttr.get(IntegerType.get_signless(64), xlu_op)
    move.attributes[CROSS_LANE_KIND] = StringAttr.get(emitter_of(xlu_op))
    move.attributes[CROSS_LANE_ROUNDS] = ArrayAttr.get(
        [DenseI64ArrayAttr.get(list(r)) for r in plan])
    return move


def _get_perm(op, name):
    return list(int_array(op.attributes[name]))


def _get_logical(shape, perm):
    out = [0] * len(shape)
    for k, a in enumerate(perm):
        out[a] = shape[k]
    return out


def _get_window(op):
    return tuple(list(int_array(op.attributes[n]))
                 for n in ("static_offsets", "static_sizes", "static_strides"))


def _get_coords(f, shape):
    x = [0] * len(shape)
    for i in reversed(range(len(shape))):
        x[i], f = f % shape[i], f // shape[i]
    return x


def _get_all_coords(shape):
    return [list(x) for x in itertools.product(*[range(n) for n in shape])]


def _get_flat(shape, perm, x):
    n = 0
    for a in perm:
        n = n * shape[a] + x[a]
    return n
