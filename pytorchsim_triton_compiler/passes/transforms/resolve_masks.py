"""pytorchsim-resolve-masks: tts -> tts. After it every access says what it moves, so convert-tts-to-pytorchsim
converts it without judging a mask: a read's element mask is a select after a box read, a write's an aliasing-aware
box read-modify-write, a gather's a select over its offsets, a box stays inside its tensor. Refuses a masked store
on a structured address that cannot be a box write and a box that starts before its tensor."""

import itertools
import math

from ...contract.stop import Reason, Refusal
from ..utils.ir_utils import (
    GATHER_MASK, GATHER_OTHER, GATHER_OFF, GATHER_PTR, GS_MASK_SEG, elem_type, funcs, gather, int_array, int_attr,
    name_of, op_of, operand_segments, splat, walk, zero_of)
from ..utils.tts_access import (
    GATHER_PTR_OP, KDYN, MAX_WRAP_PIECES, Fatal, build_read_park, build_select, get_base_offset, get_descriptor,
    get_dram_arg, get_dram_strides, get_flat_shape, get_gather_ptr_facts, get_int, get_mask_window,
    get_operand_at_segment, get_ptr_op, get_steps, is_buffer, split_access_operands)

FORBIDS = ()
MARKERS = ("tts.load", "tts.store", "tts.gather", "tts.scatter")

_GS_ATTRS = ("gather_scatter_dim", "sizes", "static_offsets",
             "static_shape", "static_strides")
_MASK_BOX = ("static_mask_dims", "static_mask_lows")
ELEMENT_MASKED = "torchsim.element_masked"
"""On the box load _rewrite_elem_mask leaves: its mask is a select after the move; the transfer's `element_masked`."""
WRAPS_PAST_THE_MASK = "torchsim.wraps_past_the_mask"
"""On an access through a make_tptr: the axes whose wrap only moves masked elements, so they are plain axes."""
BOX_FLOOR = "torchsim.box_floor"
"""On an element-masked box load: [axis, stride], the one moving axis whose positions before the tensor take the fill."""


class MaskedStoreNotABox(Fatal, Refusal):
    """A masked store on a structured address whose box cannot be read-modified-written."""
    reason = Reason.MASKED_STORE_NOT_A_BOX


class MaskedBoxBeforeTensor(Fatal, Refusal):
    """An element-masked box read of several moving axes that starts before its tensor."""
    reason = Reason.MASKED_BOX_BEFORE_TENSOR


def run(module, arg_numels=None, **opts):
    """The table of contents, in the file's order: 1 boxes over the launch, 2 boxes inside their tensors, 3 read
    masks into selects, 4 write masks into box read-modify-writes, 5 boxes over gathered rows, 6 the facts a
    structured box hands down, 7 the check that nothing is left unresolved."""
    from mlir.ir import Location

    if arg_numels is None:
        raise Fatal("arg_numels required: {arg index -> element count} for the "
                    "DRAM args (the memref<*xf32> params are unranked).")
    loc = Location.unknown()
    with loc:
        _rewrite_boxes_known_over_the_grid(module, opts.get("grid"))
        _rewrite_boxes_into_their_tensors(module, arg_numels)
        _rewrite_read_masks(module, loc)
        _rewrite_write_masks(module, arg_numels, loc)
        _rewrite_boxes_over_gathered_rows(module)
        _rewrite_structured_boxes(module, opts.get("grid"))
        _check_masks_resolved(module)
    return module


def _rewrite_boxes_known_over_the_grid(module, grid):
    """A box end the launch decides becomes static: a dim or low that keeps one value at every program id below
    the grid's bound takes it, so a numel mask that never cuts (Inductor's `x < N` where the block divides N)
    leaves no runtime clamp. An end that varies with the id, or moves with two axes, stays as it is."""
    for func in funcs(module):
        for op in list(walk(func.operation)):
            if name_of(op) in ("tts.load", "tts.store") and "static_mask_dims" in op.attributes:
                _rewrite_box_ends_known(op, grid)


def _rewrite_box_ends_known(access_op, grid):
    """Rebuild `access_op` with every box end the grid decides made static; nothing is built if none is."""
    from mlir.ir import DenseI32ArrayAttr, DenseI64ArrayAttr, InsertionPoint, Operation
    dims, lows, _, _ = split_access_operands(access_op)
    seg = operand_segments(access_op)
    at = 1 if name_of(access_op) == "tts.load" else 2
    new_dims, kept_dims = _get_ends_known(int_array(access_op.attributes["static_mask_dims"]), dims, grid, 1)
    has_lows = "static_mask_lows" in access_op.attributes
    new_lows, kept_lows = (_get_ends_known(int_array(access_op.attributes["static_mask_lows"]), lows, grid, 0)
                           if has_lows else (None, lows))
    if len(kept_dims) == len(dims) and len(kept_lows) == len(lows):
        return
    ops = list(access_op.operands)
    head, tail = ops[:sum(seg[:at])], ops[sum(seg[:at + 2]):]
    attrs = {a.name: a.attr for a in access_op.attributes}
    attrs["static_mask_dims"] = DenseI64ArrayAttr.get(new_dims)
    if has_lows:
        attrs["static_mask_lows"] = DenseI64ArrayAttr.get(new_lows)
    attrs["operandSegmentSizes"] = DenseI32ArrayAttr.get(
        list(seg[:at]) + [len(kept_dims), len(kept_lows)] + list(seg[at + 2:]))
    with InsertionPoint(access_op):
        new = Operation.create(name_of(access_op), results=[r.type for r in access_op.results],
                               operands=head + kept_dims + kept_lows + tail, attributes=attrs,
                               loc=access_op.location)
    for old, got in zip(access_op.results, new.results):
        old.replace_all_uses_with(got)
    access_op.operation.erase()


def _get_ends_known(static, values, grid, least):
    """(`static` with each dynamic end the grid decides filled in, the values still dynamic); an end below
    `least` (an empty dim) is left dynamic, for the access that never moves is not this step's to remove."""
    from ..utils.liveness import constant_over_grid
    out, kept, it = list(static), [], iter(values)
    for i, s in enumerate(static):
        if s != KDYN:
            continue
        v = next(it)
        c = constant_over_grid(v, grid)
        if c is None or c < least:
            kept.append(v)
        else:
            out[i] = c
    return out, kept


def _rewrite_boxes_into_their_tensors(module, arg_numels):
    for func in funcs(module):
        for op in walk(func.operation):
            if name_of(op) == "tts.load" and _has_element_mask(op):
                _fold_the_argument_into_the_box(op, arg_numels)


def _rewrite_read_masks(module, loc):
    from mlir.passmanager import PassManager

    changed = 0
    for func in funcs(module):
        for op in walk(func.operation):
            if op.name == "tts.gather":
                changed += _rewrite_gather(op, loc)
            elif op.name == "tts.load":
                changed += (_rewrite_elem_mask(op, loc) or _rewrite_gs_mask(op, loc))
    if changed:
        PassManager.parse("builtin.module(func.func(canonicalize,cse))").run(module.operation)


def _rewrite_write_masks(module, arg_numels, loc):
    for func in funcs(module):
        tensor_args = [a for a in func.operation.regions[0].blocks[0].arguments if is_buffer(a.type)]
        for op in [o for o in walk(func.operation) if name_of(o) == "tts.store"]:
            _rewrite_store_elem_mask(arg_numels, tensor_args, op, loc)


def _rewrite_boxes_over_gathered_rows(module):
    for func in funcs(module):
        for op in walk(func.operation):
            if name_of(op) not in ("tts.load", "tts.store"):
                continue
            gs = get_ptr_op(op)
            if name_of(gs) != GATHER_PTR_OP:
                continue
            sizes, _, _, _, _, gs_mask, _ = get_gather_ptr_facts(gs)
            if gs_mask is None:
                continue
            _refuse_read_mask_on_pointer(gs, op)
            _open_box_over_gathered(op, sizes, int_attr(gs, "gather_scatter_dim"))


def _rewrite_structured_boxes(module, grid):
    from mlir.ir import DenseI64ArrayAttr
    from ..utils.liveness import program_ids

    for func in funcs(module):
        known = program_ids(func, grid)
        for op in walk(func.operation):
            if name_of(op) not in ("tts.load", "tts.store") or name_of(get_ptr_op(op)) != "tts.make_tptr":
                continue
            ptr = get_ptr_op(op)
            sizes, strides, offsets, wrap = get_descriptor(ptr)
            sdims = {d for d, k in enumerate(get_steps(ptr)) if k}
            wdims = {d for d, w in enumerate(wrap) if not (isinstance(w, int) and w == 0)}
            past = _get_wraps_past_the_mask(op, sizes, strides, offsets, wrap, wdims - sdims)
            _check_mask_leaves_box(op, split_access_operands(op)[3])
            if past:
                op.attributes[WRAPS_PAST_THE_MASK] = DenseI64ArrayAttr.get(sorted(past))
            if wdims - past or sdims or get_flat_shape(ptr) is not None:
                continue
            base = get_base_offset(ptr)
            floor = _get_axis_below_the_tensor(
                known, op, sizes, get_dram_strides(sizes, strides),
                list(offsets) + ([] if base is None else [base]))
            if floor is not None:
                op.attributes[BOX_FLOOR] = DenseI64ArrayAttr.get(list(floor))


def _check_masks_resolved(module):
    for op in walk(module.operation):
        name = name_of(op)
        if name == "tts.load" and _has_element_mask(op):
            raise Fatal(f"an element-masked load survived _rewrite_read_masks: {op}")
        if name == "tts.store" and _has_element_mask(op) and name_of(get_ptr_op(op)) == "tts.make_tptr":
            raise Fatal(f"an element-masked store on a structured address survived: {op}")
        if name == "tts.gather":
            seg = int_array(op.attributes["operandSegmentSizes"])
            if seg[GATHER_MASK] or seg[GATHER_OTHER]:
                raise Fatal(f"a gather's mask reached convert-tts-to-pytorchsim, so _rewrite_gather "
                            f"left that carrier full: {op}")


def _has_element_mask(access_op):
    seg = operand_segments(access_op)
    return len(seg) >= 5 and bool(seg[4])


def _fold_the_argument_into_the_box(load_op, arg_numels):
    from mlir.ir import DenseI64ArrayAttr
    ptr = get_ptr_op(load_op)
    if ptr.name != "tts.make_tptr":
        return False
    sizes, strides, offsets, wrap = get_descriptor(ptr)
    if any(not isinstance(w, int) for w in wrap):
        return False
    if any(get_steps(ptr)):
        return False
    if any(not isinstance(x, int) for x in list(sizes) + list(offsets)):
        return False
    base = get_base_offset(ptr)
    if base is not None and not isinstance(base, int):
        return False
    dstride = get_dram_strides(sizes, strides)
    if any(not isinstance(d, int) or d < 0 for d in dstride):
        return False
    idx = _get_argument_index(ptr)
    numel = None if idx is None else arg_numels.get(idx)
    if numel is None:
        return False
    start = sum(offsets) + (base or 0)
    far = [max((min(n, w) if w else n) - 1, 0) * d
           for n, d, w in zip(sizes, dstride, wrap)]
    reach = sum(far)
    if start < 0 or start + reach < numel:
        return False
    cands = [a for a in range(len(sizes))
             if not wrap[a] and dstride[a] > 0 and sizes[a] > 1]
    if not cands:
        return False
    axis = max(cands, key=lambda a: dstride[a])
    room = numel - 1 - start
    if room < 0:
        return False
    fits = room // dstride[axis] + 1
    if fits >= sizes[axis]:
        return False
    dims = (list(int_array(load_op.attributes["static_mask_dims"]))
            if "static_mask_dims" in load_op.attributes else list(sizes))
    if len(dims) != len(sizes) or fits >= dims[axis]:
        return False
    dims[axis] = fits
    load_op.attributes["static_mask_dims"] = DenseI64ArrayAttr.get(dims)
    return True


def _get_argument_index(ptr_op):
    blk = op_of(ptr_op).parent
    while blk is not None and name_of(blk) != "func.func":
        blk = op_of(blk).parent
    if blk is None:
        return None
    tensors = [a for a in blk.regions[0].blocks[0].arguments if is_buffer(a.type)]
    try:
        return get_dram_arg(ptr_op.operands[0], tensors)[1]
    except Fatal:
        return None


def _rewrite_gather(op, loc):
    from mlir.ir import InsertionPoint
    seg = operand_segments(op)
    if not seg[GATHER_MASK]:
        return False
    mask = op.operands[GATHER_MASK]
    other = op.operands[GATHER_OTHER] if seg[GATHER_OTHER] else None
    res = op.results[0]
    with InsertionPoint(op):
        off = op.operands[GATHER_OFF]
        off = build_select(mask, off,
                      splat(zero_of(elem_type(off), loc), off, loc), loc)
        g = gather(op.operands[GATHER_PTR], off, res.type, loc)
        if other is None:
            other = zero_of(elem_type(g), loc)
        new = build_select(mask, g, splat(other, g, loc), loc)
    res.replace_all_uses_with(new)
    op.operation.erase()
    return True


def _rewrite_elem_mask(load_op, loc):
    from mlir.ir import DenseI32ArrayAttr, InsertionPoint, Operation, UnitAttr
    lseg = operand_segments(load_op)
    if len(lseg) < 5 or not lseg[4]:
        return False
    ops = list(load_op.operands)
    _, _, other, mask = split_access_operands(load_op)
    res = load_op.results[0]
    with InsertionPoint(load_op):
        bare = Operation.create(
            "tts.load", results=[res.type], operands=ops[:-1],
            attributes=dict(
                {"operandSegmentSizes": DenseI32ArrayAttr.get(
                    list(lseg[:4]) + [0]),
                 "static_mask_dims": load_op.attributes["static_mask_dims"],
                 ELEMENT_MASKED: UnitAttr.get()},
                **({"static_mask_lows": load_op.attributes["static_mask_lows"]}
                   if "static_mask_lows" in load_op.attributes else {})),
            loc=loc).results[0]
        if other is None:
            other = zero_of(elem_type(bare), loc)
        new = build_select(mask, bare, splat(other, bare, loc), loc)
    res.replace_all_uses_with(new)
    load_op.operation.erase()
    return True


def _rewrite_gs_mask(load_op, loc):
    from mlir.ir import DenseI32ArrayAttr, DenseI64ArrayAttr, InsertionPoint, Operation
    gs = op_of(load_op.operands[0].owner)
    if name_of(gs) != "tts.make_gather_scatter_tptr":
        return False
    gseg = operand_segments(gs)
    if not gseg[GS_MASK_SEG]:
        return False
    mask = get_operand_at_segment(gs, GS_MASK_SEG)
    dim = int_attr(gs, "gather_scatter_dim")
    sizes = int_array(gs.attributes["sizes"])
    res = load_op.results[0]

    lseg = operand_segments(load_op)
    other = list(load_op.operands)[-1] if lseg[3] else None

    with InsertionPoint(load_op):
        attrs = {n: gs.attributes[n] for n in _GS_ATTRS}
        mask_at = sum(gseg[:GS_MASK_SEG])
        attrs["operandSegmentSizes"] = DenseI32ArrayAttr.get(
            gseg[:GS_MASK_SEG] + [0] + gseg[GS_MASK_SEG + 1:])
        ops = list(gs.operands)
        idx_at = sum(gseg[:1])
        idx = ops[idx_at]
        ops[idx_at] = build_select(mask, idx, splat(_build_parked_index(gs, idx, loc), idx, loc), loc)
        bare_ptr = Operation.create(
            "tts.make_gather_scatter_tptr", results=[gs.results[0].type],
            operands=ops[:mask_at] + ops[mask_at + 1:],
            attributes=attrs, loc=loc).results[0]
        dims = int_array(load_op.attributes["static_mask_dims"])
        if dims:
            dims[dim] = sizes[dim]
        attrs = {"operandSegmentSizes": DenseI32ArrayAttr.get(lseg),
                 "static_mask_dims": DenseI64ArrayAttr.get(dims)}
        if "static_mask_lows" in load_op.attributes:
            lows = int_array(load_op.attributes["static_mask_lows"])
            if lows:
                lows[dim] = 0
            attrs["static_mask_lows"] = DenseI64ArrayAttr.get(lows)
        bare = Operation.create(
            "tts.load", results=[res.type],
            operands=[bare_ptr] + list(load_op.operands)[1:],
            attributes=attrs, loc=loc).results[0]
        if other is None:
            other = zero_of(elem_type(bare), loc)
        new = build_select(mask, bare, splat(other, bare, loc), loc, mask_axes=[dim])
    res.replace_all_uses_with(new)
    load_op.operation.erase()
    if not any(len(list(r.uses)) for r in gs.results):
        gs.erase()
    return True


def _build_parked_index(gs, idx, loc):
    """The index a masked-off row of `gs` reads, parked by build_read_park against the gather's base."""
    _, _, strides, _, _, _, base = get_gather_ptr_facts(gs)
    return build_read_park(base, strides[int_attr(gs, "gather_scatter_dim")], elem_type(idx), loc)


def _rewrite_store_elem_mask(arg_numels, tensor_args, store_op, loc):
    """An element-masked store becomes, per lap, a box read-modify-write: a lap is a run of tile
    positions inside one period of the address map, injective, so the laps run in program order and a
    later one reads what an earlier wrote. Refused (False) where the laps cannot be computed."""
    from mlir.ir import DenseI32ArrayAttr, InsertionPoint, Operation
    seg = operand_segments(store_op)
    if len(seg) < 5 or not seg[4]:
        return False
    ptr_op = op_of(store_op.operands[0].owner)
    if ptr_op is None or name_of(ptr_op) != "tts.make_tptr":
        return False
    if get_flat_shape(ptr_op) is not None:
        return False
    sizes, strides, offsets, wrap = get_descriptor(ptr_op)
    window = get_mask_window(store_op, sizes)
    laps = _get_laps(sizes, strides, wrap, window)
    if laps is None:
        return False
    bounds = _get_tile_bounds(sizes, strides, offsets, wrap,
                          get_base_offset(ptr_op))
    arg_index = get_dram_arg(ptr_op.operands[0], tensor_args)[1]
    if bounds is None or not (0 <= bounds[0]
                              and bounds[1] < arg_numels[arg_index]):
        return False
    ops = list(store_op.operands)
    box = ops[seg[0] + seg[1]:seg[0] + seg[1] + seg[2] + seg[3]]
    value, mask = ops[seg[0]], ops[-1]
    with InsertionPoint(store_op):
        for lap in laps:
            keep = {n: store_op.attributes[n] for n in _MASK_BOX if n in store_op.attributes}
            if lap:
                keep = {"static_mask_dims": _get_lap_attr(store_op, "static_mask_dims", sizes, lap, 1),
                        "static_mask_lows": _get_lap_attr(store_op, "static_mask_lows", sizes, lap, 0)}
            held = Operation.create(
                "tts.load", results=[value.type], operands=ops[:seg[0]] + box,
                attributes=dict({"operandSegmentSizes": DenseI32ArrayAttr.get(
                    [seg[0], seg[2], seg[3], 0, 0])}, **keep), loc=loc).results[0]
            Operation.create(
                "tts.store", results=[],
                operands=ops[:seg[0]] + [build_select(mask, value, held, loc)] + box,
                attributes=dict({"operandSegmentSizes": DenseI32ArrayAttr.get(
                    [seg[0], seg[1], seg[2], seg[3], 0])}, **keep), loc=loc)
    store_op.operation.erase()
    return True


def _get_laps(sizes, strides, wrap, window):
    """Each repeating axis (stride 0, or a wrap of period W/gcd(W, stride) shorter than its window) cut
    into one-period runs, one box per combination in program order; [{}] when nothing repeats. None
    where the laps are runtime, too many, or a lap still shares an address."""
    runs, spans = [], []
    for d, n in enumerate(sizes):
        st = get_int(strides[d])
        w = wrap[d] if d < len(wrap) else 0
        W = 0 if isinstance(w, int) and w == 0 else get_int(w)
        if st is None or W is None or W < 0:
            return None
        dyn = KDYN in window[d]
        lo, hi = (0, n) if dyn else (max(window[d][0], 0), min(window[d][1], n))
        period = (W // math.gcd(W, abs(st)) if st else 1) if W else (1 if st == 0 else n + 1)
        if hi - lo > period:
            if dyn:
                return None
            runs.append((d, [(a, min(a + period, hi)) for a in range(lo, hi, period)]))
        extent = min(max(hi - lo, 0), period)
        g = math.gcd(W, abs(st)) if W else abs(st)
        spans.append((g, W - g if W else (extent - 1) * g, extent))
    reach = 0
    for g, span, extent in sorted(spans):
        if extent <= 1:
            continue
        if g <= reach:
            return None
        reach += span
    total = 1
    for _, chunks in runs:
        total *= len(chunks)
    if total > MAX_WRAP_PIECES:
        return None
    return [{d: c for (d, _), c in zip(runs, combo)}
            for combo in itertools.product(*[c for _, c in runs])]


def _get_lap_attr(access_op, attr, sizes, lap, end):
    """`attr` with the lap's axes replaced by its (low, high)[`end`]; the others keep what the access said."""
    from mlir.ir import DenseI64ArrayAttr
    old = int_array(access_op.attributes[attr]) if attr in access_op.attributes else []
    if len(old) != len(sizes):
        old = list(sizes) if end else [0] * len(sizes)
    return DenseI64ArrayAttr.get([lap[d][end] if d in lap else v for d, v in enumerate(old)])


def _get_tile_bounds(sizes, strides, offsets, wrap, base):
    lo = hi = 0
    if base is not None:
        o = get_int(base)
        if o is None:
            return None
        lo += o
        hi += o
    for d in range(len(sizes)):
        w = wrap[d] if d < len(wrap) else 0
        if not (isinstance(w, int) and w == 0):
            W = get_int(w)
            if W is None or W <= 0:
                return None
            g, o = math.gcd(W, abs(get_int(strides[d]) or 0)), get_int(offsets[d])
            hi += W - g + o % g if g and o is not None else W - 1
            continue
        n, st, off = (get_int(sizes[d]), get_int(strides[d]),
                      get_int(offsets[d]))
        if None in (n, st, off):
            return None
        lo += off
        hi += off
        span = (n - 1) * st
        hi += span if span >= 0 else 0
        lo += span if span < 0 else 0
    return lo, hi


def _refuse_read_mask_on_pointer(gs_op, access_op):
    if name_of(access_op) == "tts.load":
        raise Fatal(f"a read's mask reached the pointer of a gather-scatter, so _rewrite_read_masks "
                    f"left that carrier full: {gs_op}")


def _open_box_over_gathered(access_op, sizes, gather_dim):
    from mlir.ir import DenseI64ArrayAttr
    if "static_mask_dims" not in access_op.attributes:
        return
    dims = int_array(access_op.attributes["static_mask_dims"])
    if not dims:
        return
    dims[gather_dim] = sizes[gather_dim]
    access_op.attributes["static_mask_dims"] = DenseI64ArrayAttr.get(dims)
    if "static_mask_lows" in access_op.attributes:
        lo = int_array(access_op.attributes["static_mask_lows"])
        lo[gather_dim] = 0
        access_op.attributes["static_mask_lows"] = DenseI64ArrayAttr.get(lo)


def _check_mask_leaves_box(access_op, elem_mask):
    if elem_mask is not None:
        raise MaskedStoreNotABox(
            f"a masked store on a structured address cannot be a box write: a box "
            f"read-modify-write needs distinct, in-bounds, unwrapped addresses, and "
            f"a mask never turns it into a per-element transfer: {access_op}")


def _get_wraps_past_the_mask(access_op, sizes, strides, offsets, wrap, wdims):
    """The wrapping axes whose every element inside the access's static mask sits in the wrap's first lap: the
    wrap moves only masked elements there, so the axis is a plain one (a head index (k // D) % H under a K tile
    wider than H * D)."""
    if "static_mask_dims" not in access_op.attributes:
        return set()
    hi = int_array(access_op.attributes["static_mask_dims"])
    lo = (int_array(access_op.attributes["static_mask_lows"])
          if "static_mask_lows" in access_op.attributes else [0] * len(hi))
    if len(hi) != len(sizes) or len(lo) != len(sizes):
        return set()
    past = set()
    for d in wdims:
        n, st = get_int(sizes[d]), get_int(strides[d])
        W, o = get_int(wrap[d]), get_int(offsets[d])
        if None in (n, st, W, o) or W <= 0 or st < 0 or KDYN in (lo[d], hi[d]):
            continue
        first, last = max(lo[d], 0), min(hi[d], n) - 1
        if first <= last and 0 <= o + first * st and o + last * st < W:
            past.add(d)
    return past


def _get_axis_below_the_tensor(known, access_op, sizes, dstride, offsets):
    """(axis, stride) of an element-masked box whose start (its offsets plus its box's static lows) may lie
    before its tensor and whose one moving axis can be clamped exactly; else None."""
    from ..utils.liveness import interval
    if ELEMENT_MASKED not in access_op.attributes:
        return None
    lo = 0
    for o in offsets:
        got = (o, o) if isinstance(o, int) else interval(o, known)
        if got is None:
            lo = None
            break
        lo += got[0]
    if lo is not None:
        lo += sum(low * dstride[d] for d, (low, _) in enumerate(get_mask_window(access_op, sizes))
                  if low != KDYN and low > 0 and isinstance(dstride[d], int) and dstride[d] > 0)
    if lo is not None and lo >= 0:
        return None
    moving = [d for d, n in enumerate(sizes) if n > 1 and not (isinstance(dstride[d], int) and dstride[d] == 0)]
    if len(moving) != 1 or not isinstance(dstride[moving[0]], int) or dstride[moving[0]] < 0:
        if lo is not None:
            raise MaskedBoxBeforeTensor(
                f"an element-masked read of {len(moving)} moving axes may start {-lo} elements before its tensor: "
                f"clamping an axis would drop the valid elements of the rows it cuts, "
                f"and a masked-off element must not be read: {access_op}")
        return None
    return moving[0], dstride[moving[0]]
