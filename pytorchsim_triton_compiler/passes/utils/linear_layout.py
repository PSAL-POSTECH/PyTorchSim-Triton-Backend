"""A tile's layout as TritonGPU's LinearLayout gives it -- register and lane bases, each a logical
coordinate -- and the one physical order it has here: the lane axis, then the register bits slowest
first, so an element's place is its (lane, register) number."""
from ...dialect.pytorchsim import reshape_layout
from .ir_utils import opaque_body_attr


def get_bases(enc):
    """(register bases, lane bases) of a `#ttg.linear` with no warp or block bases, or None."""
    body = opaque_body_attr(enc, "ttg", "linear")
    if body is None:
        return None
    read = lambda key: [[int(x.value) for x in basis] for basis in body[key]]
    if read("warp") or read("block"):
        return None
    return read("register"), read("lane")


def _bit_of(basis):
    moved = [(d, v) for d, v in enumerate(basis) if v]
    if not moved:
        return None
    if len(moved) > 1 or moved[0][1] & (moved[0][1] - 1):
        return False
    return moved[0][0], moved[0][1].bit_length() - 1


def physical_perm(bases, shape):
    """`new_shape[i] = shape[p[i]]` that lays the tile out as its bases do, or WHY NOT. The lane
    dim goes first; a replica (no lane bits) puts a unit dim there for its one bank, and a unit dim
    before the lane dim follows it, so a reduce of the lanes leaves [unit, depth...] in order."""
    if not shape:
        return []
    regs, lanes = bases
    lane_bits = [b for b in (_bit_of(x) for x in lanes) if b is not None]
    reg_bits = [b for b in (_bit_of(x) for x in regs) if b is not None]
    if False in lane_bits or False in reg_bits:
        return "a basis that steps more than one bit"
    lane_dims = {d for d, _ in lane_bits}
    if len(lane_dims) > 1:
        return "lanes over several dims"
    lane = next(iter(lane_dims)) if lane_dims else None
    if lane is not None and [b for _, b in lane_bits] != list(range(len(lane_bits))):
        return "lane bits that are not the dim's lowest, in order"
    wrapped = [b for d, b in reg_bits if d == lane]
    if wrapped and (len(lane_bits) != len(lanes)
                    or reg_bits[-len(wrapped):] != [(lane, b) for b in wrapped]
                    or sorted(wrapped) != list(range(len(lane_bits), len(lane_bits) + len(wrapped)))):
        return "a dim split between the lanes and the registers"
    order = []
    for d, b in reversed([x for x in reg_bits if x[0] != lane]):
        if order and order[-1][0] == d:
            if order[-1][1] != b + 1:
                return "a dim's register bits out of order"
            order[-1] = (d, b)
        elif any(o == d for o, _ in order):
            return "a dim's register bits interleaved with another's"
        else:
            order.append((d, b))
    depth = [d for d, _ in order]
    units = [d for d, n in enumerate(shape) if d != lane and d not in depth]
    if lane is None:
        if not units:
            return "a replica with no unit dim to stand for its one bank"
        return [units[0]] + depth + units[1:]
    before = [d for d in units if d < lane]
    return [lane] + before + depth + [d for d in units if d > lane]


def compose_reshape(bases, src_shape, dst_shape):
    """The bases after a row-major reshape -- each coordinate renamed, none moved -- as Triton's
    reshapeLayout gives them; a dim kept whole goes through at its power-of-two extent, and any other
    dim not a power of two is refused (ValueError)."""
    src, dst = _whole_at_power_of_two(list(src_shape), list(dst_shape))
    got = dict(reshape_layout([("register", bases[0]), ("lane", bases[1])], src, dst))
    return got["register"], got["lane"]


def _whole_at_power_of_two(src, dst):
    """(src, dst) with each dim the reshape keeps whole -- one src dim that is one dst dim of its size,
    between two row-major group edges -- rounded up to a power of two; a 31-wide slice has a 32's bases."""
    i = j = 0
    while i < len(src) and j < len(dst):
        a, b, i0, j0 = src[i], dst[j], i, j
        i, j = i + 1, j + 1
        while a != b:
            if (i == len(src)) if a < b else (j == len(dst)):
                raise ValueError(f"a reshape of {src} to {dst} does not keep the elements")
            if a < b:
                a, i = a * src[i], i + 1
            else:
                b, j = b * dst[j], j + 1
        if i - i0 == 1 and j - j0 == 1 and src[i0] & (src[i0] - 1):
            src[i0] = dst[j0] = 1 << (src[i0] - 1).bit_length()
    return src, dst


def spell(bases):
    """`#ttg.linear` text of (register bases, lane bases), no warp or block."""
    show = lambda bs: "[" + ", ".join("[" + ", ".join(str(x) for x in b) + "]" for b in bs) + "]"
    return "#ttg.linear<{register = %s, lane = %s, warp = [], block = []}>" % (show(bases[0]), show(bases[1]))


def drop_dims(bases, dims):
    """The bases once `dims` are gone -- a reduce over them, or a unit dim collapsed."""
    keep = lambda b: [x for d, x in enumerate(b) if d not in dims]
    return [keep(b) for b in bases[0]], [keep(b) for b in bases[1]]


def without_idle_registers(bases):
    """The bases with every register basis that steps nothing dropped: the form regularize-layouts gives
    every layout, which a pass building one keeps."""
    return [b for b in bases[0] if any(b)], bases[1]


def cut_lanes(bases, dim, extent):
    """The bases of the first `extent` positions along `dim` (a power of two): a basis stepping
    `dim` past them steps nothing."""
    cut = lambda b: [0 if d == dim and x >= extent else x for d, x in enumerate(b)]
    return [cut(b) for b in bases[0]], [cut(b) for b in bases[1]]


def digit_bounds(bases, shape):
    """For each dim, the bit positions where its run changes -- lane to register, or one register
    run to another -- as a sorted list inside (0, log2 extent): cut there, every piece (digit) is
    wholly lanes or wholly one run of registers."""
    regs, lanes = bases
    owner = {}
    for kind, bs in (("lane", lanes), ("reg", regs)):
        for i, b in enumerate(bs):
            got = _bit_of(b)
            if got is False:
                return None
            if got is not None:
                owner[got] = (kind, i)
    out = []
    for d, n in enumerate(shape):
        width = max(n - 1, 0).bit_length()
        cuts = []
        for bit in range(1, width):
            a, b = owner.get((d, bit - 1)), owner.get((d, bit))
            if a is None or b is None or a[0] != b[0] or a[1] + 1 != b[1]:
                cuts.append(bit)
        out.append(cuts)
    return out


def widen(bases, dim, extent):
    """The bases of a tile grown along `dim` to `extent` (rounded up to a power of two, the hull):
    each new bit of `dim` the next idle lane where the dim rides the lanes, else the next register
    -- the tile laid out as its source, with more of the same dim."""
    regs, lanes = [list(b) for b in bases[0]], [list(b) for b in bases[1]]
    steps = lambda bs: [b[dim] for b in bs if b[dim] and not any(x for d, x in enumerate(b) if d != dim)]
    have = steps(regs) + steps(lanes)
    top = max(have) if have else 0
    on_lanes = bool(steps(lanes)) and (not steps(regs) or max(steps(lanes)) == top)
    bit = top * 2 if top else 1
    while bit < extent:
        idle = next((b for b in lanes if not any(b)), None) if on_lanes else None
        if idle is not None:
            idle[dim] = bit
        else:
            at = max((i + 1 for i, b in enumerate(regs) if b[dim]), default=len(regs))
            regs.insert(at, [bit if d == dim else 0 for d in range(len(regs[0]) if regs else len(lanes[0]))])
        bit *= 2
    return regs, lanes


def lanes_on_one_dim(bases):
    """Bases whose lanes step several dims, laid with the lanes on the dim that holds the most lane
    bits (the lower dim on a tie) and every other lane bit in the registers, slowest first: the tile
    this machine can hold, one move away from the one TritonGPU wrote."""
    regs, lanes = bases
    bits = [(_bit_of(b), b) for b in lanes]
    count = {}
    for got, _ in bits:
        if got:
            count[got[0]] = count.get(got[0], 0) + 1
    keep = min(count, key=lambda d: (-count[d], d))
    new_lanes, moved = [], []
    for got, b in bits:
        if got and got[0] != keep:
            moved.append(list(b))
            new_lanes.append([0] * len(b))
        else:
            new_lanes.append(list(b))
    return [list(b) for b in regs] + moved, new_lanes
