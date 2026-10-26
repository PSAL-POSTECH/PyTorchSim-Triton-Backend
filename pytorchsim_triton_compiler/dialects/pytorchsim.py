"""The `pytorchsim` dialect: the python extension it lives in (shared with dialects/togsim.py,
dialects/pytorchsim_tpu.py, dialects/pytorchsim_rvv.py and dialects/pytorchsim_extern.py), the op
classes ODS generates from the same .td, and what ODS cannot state -- which of a transfer's attributes
speak per axis, and the `dram_arg` values that name no argument."""
import glob
import hashlib
import importlib.util
import os
import sys
from collections import namedtuple

BUILD = os.path.join(os.environ.get("PYTORCHSIM_PREFIX", "/workspace"), "pytorchsim-dialect-build")

_done = [False]
_ext = [None]


def _load_from_build(name, pattern):
    hits = sorted(glob.glob(os.path.join(BUILD, pattern)))
    if not hits:
        raise ImportError(
            f"{os.path.join(BUILD, pattern)} is missing: the dialect build at {BUILD} "
            f"did not produce it: `setup/toolchain.sh dialect`.")
    spec = importlib.util.spec_from_file_location(f"{__package__}.{name}", hits[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_generated(name):
    return _load_from_build(name, name + ".py")


def _refuse_a_stale_build():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "include", "PyTorchSim")
    want = {}
    for name in ("PyTorchSimOps.td", "PyTorchSimDialect.td", "TogsimOps.td", "PyTorchSimTpuOps.td",
                 "PyTorchSimExternOps.td", "PyTorchSimRvvOps.td"):
        path = os.path.join(src, name)
        if not os.path.exists(path):
            return
        want[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    try:
        got = _load_generated("_pytorchsim_stamp").TD_SHA256
    except ImportError:
        raise ImportError(
            f"the dialect build at {BUILD} carries no .td stamp, so nothing says which "
            f"schema its op classes came from. Rebuild it -- `ninja -C {BUILD}`.") from None
    stale = sorted(n for n in want if want[n] != got.get(n))
    if stale:
        raise ImportError(
            f"the dialect build at {BUILD} was generated from a different "
            f"{', '.join(stale)}: its op classes and operand table are that .td's, not "
            f"this checkout's, and an op built from them can be silently wrong. "
            f"Rebuild it -- `ninja -C {BUILD}`.")


_refuse_a_stale_build()
_load_generated("_pytorchsim_ops_gen")
from ._pytorchsim_ops_gen import *  # noqa: E402,F401,F403
from ._pytorchsim_ops_gen import RuntimeAttributeOp, TransferOp, WaitOp  # noqa: E402,F401
_TABLE = _load_generated("_pytorchsim_operands")


def register():
    """Put `pytorchsim`, `pytorchsim_tpu`, `pytorchsim_rvv`, `pytorchsim_extern` and `togsim` in the
    bindings' GLOBAL registry, once, the passes in the pass registry, and let `.opview` find the
    classes above and the sibling modules'."""
    if _done[0]:
        return
    from mlir._mlir_libs import _mlir, get_dialect_registry
    ext = _load_from_build("_pytorchsim_ext", "_pytorchsim_ext*.so")
    ext.insert_into_registry(get_dialect_registry())
    ext.register_passes()
    _ext[0] = ext
    _mlir.globals.append_dialect_search_prefix(__package__)
    _done[0] = True


EFFECTS = ("read", "write", "allocate", "free", "sync")
"""What effects_on and effects_of answer, in the bit order _pytorchsim_ext states them; "sync" is an effect on
the transfers in flight into or out of a buffer (pytorchsim::InFlight), not on its data."""
UNKNOWN = frozenset({"unknown"})


def effects_on(op, value):
    """Which of EFFECTS `op` states it has on `value`, off MemoryEffectOpInterface; UNKNOWN when
    `op` states no effects at all."""
    bits = _ext[0].effects_on(getattr(op, "operation", op), value)
    if bits < 0:
        return UNKNOWN
    return frozenset(name for k, name in enumerate(EFFECTS) if bits >> k & 1)


def effects_of(op):
    """Which of EFFECTS `op` states on any value or resource, its regions' included; UNKNOWN when
    it or an op in it states no effects at all."""
    bits = _ext[0].effects_of(getattr(op, "operation", op))
    if bits < 0:
        return UNKNOWN
    return frozenset(name for k, name in enumerate(EFFECTS) if bits >> k & 1)


#: A pack's own dialect is `pytorchsim_<family>`: its ops are lowered by that pack alone.
TARGET_DIALECT_PREFIX = "pytorchsim_"
#: The ops a kernel writes itself: the same prefix, and no pack's.
EXTERN_DIALECT = "pytorchsim_extern"


def is_target_op(op):
    """Is `op` of a pack's own dialect?"""
    dialect = op.name.split(".", 1)[0]
    return dialect.startswith(TARGET_DIALECT_PREFIX) and dialect != EXTERN_DIALECT


def is_lane_variant(op):
    """Whether `op` declares the dialect's LaneVariant trait: its result differs between lanes
    though no operand of it and no bank it reads says so."""
    return _ext[0].is_lane_variant(getattr(op, "operation", op))


def is_view_like(op):
    """Whether `op` is a ViewLikeOpInterface op: its result is a view of its source buffer."""
    return _ext[0].is_view_like(getattr(op, "operation", op))


def is_bank_view(op):
    """Whether `op` declares the dialect's BankView trait: its source as one lane's bank, its result
    type's strided layout all it states."""
    return _ext[0].is_bank_view(getattr(op, "operation", op))


def is_bank_words(op):
    """Whether `op` declares the dialect's BankWords trait: a bank read as another element type in
    the same descriptor."""
    return _ext[0].is_bank_words(getattr(op, "operation", op))


def is_memory_effect_free(op):
    """Whether `op` states no memory effect on any value or resource, its regions' included."""
    return _ext[0].is_memory_effect_free(getattr(op, "operation", op))


Loop = namedtuple("Loop", "ivs lbs ubs steps iter_args")
"""What LoopLikeOpInterface states of one loop: its ivs, its bounds (a Value or an int each), and
(iter_arg, init, yielded or None) per iter_arg."""


def loop_like(op):
    """The Loop `op` is, or None when it is no LoopLikeOpInterface loop that states its ivs."""
    o = getattr(op, "operation", op)
    ivs = _ext[0].loop_ivs(o)
    if ivs is None:
        return None
    return Loop(list(ivs), list(_ext[0].loop_bounds(o, 0)), list(_ext[0].loop_bounds(o, 1)),
                list(_ext[0].loop_bounds(o, 2)), list(_ext[0].loop_iter_args(o) or []))


def branch_controls(op):
    """The operands of a RegionBranchOpInterface op that is no loop which pick the region it runs,
    or None when `op` is no such branch."""
    got = _ext[0].branch_controls(getattr(op, "operation", op))
    return None if got is None else list(got)


def is_loop(op):
    """Whether `op` is a LoopLikeOpInterface loop, one that states ivs (loop_like) or not."""
    return bool(_ext[0].is_loop(getattr(op, "operation", op)))


def region_inputs(arg):
    """The values that flow into `arg`, a block argument of a region of a RegionBranchOpInterface op,
    from the op's entry and its terminators; None when `arg` is no such argument."""
    got = _ext[0].region_inputs(arg)
    return None if got is None else list(got)


def terminator_controls(op):
    """The operands of a region terminator forwarded to no successor -- a while's condition -- or
    None when `op` is no RegionBranchTerminatorOpInterface op."""
    got = _ext[0].terminator_controls(getattr(op, "operation", op))
    return None if got is None else list(got)


def signed_ranges(scope, values):
    """(smin, smax) of each of `values` as upstream's integer range analysis over `scope` proves it,
    None where it proves nothing; the ops of this dialect state theirs (program_id, lane_id)."""
    got = _ext[0].signed_ranges(getattr(scope, "operation", scope), list(values))
    if got is None:
        raise RuntimeError(f"the integer range analysis failed over {getattr(scope, 'name', scope)}")
    return list(got)


def has_no_signed_wrap(op):
    """Whether `op` states nsw in its overflow flags (arith's ArithIntegerOverflowFlagsInterface)."""
    got = _ext[0].has_no_signed_wrap(getattr(op, "operation", op))
    if got < 0:
        raise TypeError(f"{op.name} states no overflow flags")
    return bool(got)


def mark_no_signed_wrap(op):
    """Add nsw to `op`'s overflow flags (arith's ArithIntegerOverflowFlagsInterface)."""
    if _ext[0].mark_no_signed_wrap(getattr(op, "operation", op)) < 0:
        raise TypeError(f"{op.name} states no overflow flags")


def view_root(value):
    """The buffer `value` is a view of, through every ViewLikeOpInterface in its memory space."""
    return _ext[0].view_root(value)


def reshape_layout(bases, src_shape, dst_shape):
    """[(in-dim, bases)] of a layout over `src_shape` once reshaped row-major to `dst_shape`: Triton's
    reshapeLayout, which refuses (ValueError) a shape not all powers of two."""
    return _ext[0].reshape_layout(bases, list(src_shape), list(dst_shape))


def operand_number(op, name, k=0):
    """Where the `k`-th operand of ODS argument `name` sits among `op`'s operands: the .td's order,
    and the op's own `operandSegmentSizes` where it has one."""
    from mlir.ir import DenseI32ArrayAttr
    op = getattr(op, "operation", op)
    i = _TABLE.OPERANDS[op.name].index(name)
    if "operandSegmentSizes" not in op.attributes:
        return i + k
    sizes = list(DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
    return sum(sizes[:i]) + k


def restated(op, operands=None, attributes=None):
    """A new `op` at the insertion point: each ODS argument `operands` names given those values, the
    attributes updated by `attributes` (None removes one), the segment sizes restated to match."""
    from mlir.ir import DenseI32ArrayAttr, Operation
    op = getattr(op, "operation", op)
    names = _TABLE.OPERANDS[op.name]
    sizes = list(DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
    flat, groups = list(op.operands), []
    for k in sizes:
        groups.append(flat[:k])
        flat = flat[k:]
    for name, vals in (operands or {}).items():
        groups[names.index(name)] = list(vals)
    attrs = {a.name: a.attr for a in (op.attributes[k] for k in range(len(op.attributes)))}
    for name, attr in (attributes or {}).items():
        if attr is None:
            attrs.pop(name, None)
        else:
            attrs[name] = attr
    attrs["operandSegmentSizes"] = DenseI32ArrayAttr.get([len(g) for g in groups])
    return Operation.create(op.name, results=[r.type for r in op.results],
                            operands=[v for g in groups for v in g], attributes=attrs,
                            loc=op.location)


def tile_shape(sram_type):
    """The tile a scratchpad buffer of `sram_type` holds, a scalar's being one element -- the one
    definition of that, which a transfer's descriptor and its TOG node both state."""
    from mlir.ir import ShapedType
    return list(ShapedType(sram_type).shape) or [1]


def walked_shape(transfer):
    """The shape one transfer's descriptor walks: its `subtile_size` if it has one, else its
    scratchpad tile -- a `piece_box` narrows what it moves, not what it walks."""
    from mlir.ir import ArrayAttr, IntegerAttr
    op = getattr(transfer, "operation", transfer)
    if "subtile_size" in op.attributes:
        return [IntegerAttr(a).value for a in ArrayAttr(op.attributes["subtile_size"])]
    return tile_shape(op.opview.sram.type)


def moved_shape(transfer):
    """The shape one transfer moves: what it walks, narrowed to its `piece_box`.
    prod(moved_shape) is what it signals its semaphore with -- the one definition of that."""
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr
    op = getattr(transfer, "operation", transfer)
    shape = walked_shape(op)
    if "piece_box" in op.attributes:
        for d, entry in enumerate(ArrayAttr(op.attributes["piece_box"])):
            lohi = list(DenseI64ArrayAttr(entry))
            if lohi:
                shape[d] = lohi[1] - lohi[0]
    return shape


def is_load(transfer):
    """Whether a transfer moves DRAM into the scratchpad: the one definition every pass shares, so
    a load and a store are told apart in a single place."""
    return is_load_kind(getattr(transfer, "operation", transfer).opview.direction.value)


def is_load_kind(direction):
    """Whether a `direction` value names a load -- for a transfer not built yet; else is_load."""
    return direction == "load"


def stated_attributes(op_name):
    """Every attribute ODS lets `op_name` state of itself, with the segment sizes a variadic op
    carries -- what is NOT one of these was put there by a pass."""
    return frozenset(_TABLE.ATTRIBUTES[op_name]) | {"operandSegmentSizes"}


AXIS_NUMBERS = "a list of axis numbers"
AXIS_NUMBER = "one axis number"
PER_TILE_AXIS = "one entry per tile axis"
DIGITS_PER_TILE_AXIS = "one digit list per tile axis"

Digit = namedtuple("Digit", "radix stride modulus phase")
"""One digit of a `cut` axis: its radix (0 for the outermost, the rest of the axis), the DRAM
stride a unit of it steps, and a modulus W > 0 with the phase it starts at when it wraps."""
CUT_DIGIT_FIELDS = len(Digit._fields)


def cut_digits(entry):
    """One `cut` entry -- a flat list of digit records, outer to inner -- as its Digits."""
    vals = list(entry)
    return tuple(Digit(*vals[k:k + CUT_DIGIT_FIELDS]) for k in range(0, len(vals), CUT_DIGIT_FIELDS))


def cut_entry(digits):
    """The `cut` entry stating `digits`: their records flattened, outer to inner."""
    return [x for d in digits for x in d]


def cut_offset(digits, i):
    """The DRAM offset, from dram_idx, of coordinate `i` of an axis cut into `digits`: `i` split
    mixed-radix, a digit j adding j*stride, or (phase + j*stride) mod W - phase when it wraps."""
    total = 0
    for d in reversed(digits):
        j, i = (i % d.radix, i // d.radix) if d.radix else (i, 0)
        total += ((d.phase + j * d.stride) % d.modulus - d.phase if d.modulus
                  else j * d.stride)
    return total

def cut_translation(digits, step, count):
    """The DRAM delta D with cut_offset(digits, k*step + c) == k*D + cut_offset(digits, c) for every
    k < count and c < step: what moving an axis's start by `step` adds to dram_idx; None if none."""
    first = [cut_offset(digits, c) for c in range(step)]
    delta = cut_offset(digits, step) - first[0] if count > 1 else 0
    if all(cut_offset(digits, k * step + c) == k * delta + first[c]
           for k in range(1, count) for c in range(step)):
        return delta
    return None


PER_AXIS = {
    "dram_stride": PER_TILE_AXIS,
    "masked_axes": AXIS_NUMBERS,
    "dyn_stride_axes": AXIS_NUMBERS,
    "indirect_dim": AXIS_NUMBER,
    "subtile_size": PER_TILE_AXIS,
    "cut": DIGITS_PER_TILE_AXIS,
    "piece_box": PER_TILE_AXIS,
}
_unknown = set(PER_AXIS) - set(_TABLE.ATTRIBUTES[TransferOp.OPERATION_NAME])
if _unknown:
    raise ImportError(f"PER_AXIS names {sorted(_unknown)}, which the transfer's "
                      f"ODS does not declare")



def renumbered_per_axis(op, groups, shape):
    """`op`'s per-axis attributes restated, without touching `op`, for new axes each a group of old
    ones: a per-tile entry is the group's last axis of extent > 1 (its first if none), as collapse_shape
    strides it, a cut its movers' digits joined; an axis named by number must be its group's only mover."""
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr, IntegerAttr, IntegerType
    i64 = IntegerType.get_signless(64)
    rep = [next((d for d in reversed(g) if shape[d] != 1), g[0]) for g in groups]
    alone = {d: k for k, g in enumerate(groups) for d in g
             if all(shape[e] == 1 for e in g if e != d)}

    def axis(name, d):
        if d not in alone:
            raise NotImplementedError(
                f"{name} names tile axis {d}, which this rewrite peels, drops or folds with "
                f"another axis that moves, so no new axis is axis {d} alone: {op}")
        return alone[d]

    out = {}
    for name, kind in PER_AXIS.items():
        if name not in op.attributes:
            continue
        got = op.attributes[name]
        if kind == AXIS_NUMBERS:
            out[name] = ArrayAttr.get(
                [IntegerAttr.get(i64, axis(name, IntegerAttr(a).value)) for a in ArrayAttr(got)])
        elif kind == AXIS_NUMBER:
            out[name] = IntegerAttr.get(i64, axis(name, IntegerAttr(got).value))
        elif kind == DIGITS_PER_TILE_AXIS:
            continue
        elif kind == PER_TILE_AXIS:
            try:
                vals = list(DenseI64ArrayAttr(got))
                out[name] = DenseI64ArrayAttr.get([vals[r] for r in rep])
            except (TypeError, ValueError):
                items = list(ArrayAttr(got))
                out[name] = ArrayAttr.get([items[r] for r in rep])
        else:
            raise RuntimeError(f"a per-axis kind nothing renumbers: {name}: {kind}")
    _renumber_cut(op, groups, shape, out)
    return out


def _renumber_cut(op, groups, shape, out):
    """The `cut` of the new axes into `out`, None when no new axis is cut: each new axis's digits
    are its old movers' concatenated (_collapsed_digits), a plain stride being one affine digit."""
    from mlir.ir import ArrayAttr, DenseI64ArrayAttr, IntegerAttr
    rank = max(len(shape), 1)
    cut = ([cut_digits(DenseI64ArrayAttr(a)) for a in ArrayAttr(op.attributes["cut"])]
           if "cut" in op.attributes else [()] * rank)
    stride = [IntegerAttr(a).value for a in ArrayAttr(op.attributes["dram_stride"])]
    entries = [_collapsed_digits(op, g, shape, stride, cut) for g in groups]
    if any(entries):
        out["cut"] = ArrayAttr.get([DenseI64ArrayAttr.get(cut_entry(e)) for e in entries])
    elif "cut" in op.attributes:
        out["cut"] = None


def _collapsed_digits(op, g, shape, stride, cut):
    """The digits of one new axis standing for old axes `g`: each mover's (its outermost radix
    made its share of its extent), outer to inner, DRAM-contiguous neighbours merged; () when
    that leaves one affine digit. A cut unit axis folded with another must add 0 at its coordinate."""
    import math
    movers = [d for d in g if shape[d] != 1]
    if len(g) > 1 and any(cut[d] and cut_offset(cut[d], 0) for d in g if d not in movers):
        raise NotImplementedError(
            f"a unit axis whose cut offsets its only coordinate is folded with another, and the "
            f"new axis's digits cannot state that offset: {op}")
    if not movers:
        return cut[g[0]]
    out = []
    for k, d in enumerate(movers):
        ds = list(cut[d]) or [Digit(0, stride[d], 0, 0)]
        if k:
            inner = math.prod(x.radix for x in ds[1:])
            if shape[d] % inner:
                raise NotImplementedError(
                    f"tile axis {d} of extent {shape[d]} is cut into inner digits of "
                    f"{inner}, which do not divide it, so it is no digit of the new axis: {op}")
            ds[0] = ds[0]._replace(radix=shape[d] // inner)
        out += ds
    merged = [out[0]]
    for x in out[1:]:
        p = merged[-1]
        if not p.modulus and not x.modulus and p.stride == x.radix * x.stride:
            merged[-1] = Digit(p.radix * x.radix, x.stride, 0, 0)
        else:
            merged.append(x)
    return () if len(merged) == 1 and not merged[0].modulus else tuple(merged)


def renumber_per_axis(op, groups, shape):
    """Write renumbered_per_axis(op, groups, shape) onto `op`, removing what it drops."""
    for name, attr in renumbered_per_axis(op, groups, shape).items():
        if attr is None:
            del op.attributes[name]
        else:
            op.attributes[name] = attr


DRAM_ARG_POINTER_TILE = -2

POINTER_TILE_BASE = 8
"""Where a DRAM_ARG_POINTER_TILE transfer counts its indices from. NOT 0: the DMA reads a
DRAM address of 0 as "outside the box, fill"; 8 is past it and a multiple of every width."""


register()
