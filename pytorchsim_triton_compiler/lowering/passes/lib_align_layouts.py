"""lib_align_layouts: upstream's ShapeUtil::AlignLayouts, the rule that says when a reshape is
a bitcast in the orders its two ends hold. A RULE TAKEN FROM XLA, not its assignment: the
orders come from the TritonGPU encodings, this only answers whether a reshape moves data."""

from .lib_ir import CommonFactors


class Layout:
    """Upstream's `Layout`: the dim order, minor first. KEPT AS A VECTOR even
    though this machine rides one axis, because a vector is what says which of
    two tiles is the same tile -- an axis alone cannot."""

    def __init__(self, minor_to_major=None):
        self.minor_to_major_ = list(minor_to_major or [])

    def minor_to_major(self):
        return self.minor_to_major_

    def __eq__(self, other):
        """UPSTREAM'S `Layout::Equal().MinorToMajorOnly()`. Without it `==` is
        identity, every equal layout reads as a change, and any cycle in the
        constraint graph propagates for ever."""
        return (isinstance(other, Layout)
                and self.minor_to_major_ == other.minor_to_major_)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(self.minor_to_major_))

    def __repr__(self):
        return "{%s}" % ",".join(str(d) for d in self.minor_to_major_)


def _AlignMinorToMajor(dims, minor_to_major, other):
    """Upstream's `ShapeUtil::AlignLayouts` on physical orders: `other`'s minor_to_major that makes
    the reshape a bitcast, or None. Unit axes go just under the lane, not above it as upstream's do."""
    if 1 in dims or 1 in other:
        keep = [k for k, d in enumerate(dims) if d != 1]
        renumber = {old: new for new, old in enumerate(keep)}
        got = _AlignMinorToMajor([dims[k] for k in keep],
                                 [renumber[d] for d in minor_to_major if d in renumber],
                                 [d for d in other if d != 1])
        if got is None:
            return None
        dim_map = [k for k, d in enumerate(other) if d != 1]
        got = [dim_map[d] for d in got]
        ones = [k for k in reversed(range(len(other))) if other[k] == 1]
        return got[:-1] + ones + got[-1:] if got else ones
    factors = CommonFactors(dims, other)
    to_factor = [0] * len(dims)
    for pos in range(len(factors) - 1):
        start, end = factors[pos][0], factors[pos + 1][0]
        if start == end:
            continue
        physical = minor_to_major.index(start)
        to_factor[start] = pos
        for d in range(start + 1, end):
            physical -= 1
            if physical < 0 or minor_to_major[physical] != d:
                return None
            to_factor[d] = pos
    out, minor = [], 0
    while len(out) < len(other):
        f = to_factor[minor_to_major[minor]]
        (s0, s1), (e0, e1) = factors[f], factors[f + 1]
        out.extend(range(e1 - 1, s1 - 1, -1))
        minor += e0 - s0
    return out


def AlignLayouts(shape, layout, other):
    """Upstream's `ShapeUtil::AlignLayouts`: the order of `other` that makes the reshape from `shape`
    a bitcast, or None. A bitcast here also keeps the lanes, so both ends lead with one extent."""
    order = layout.minor_to_major() if layout is not None else []
    if sorted(order) != list(range(len(shape))):
        return None
    got = _AlignMinorToMajor(list(shape), PhysicalMinorToMajor(layout), list(other))
    if got is None:
        return None
    lead = got[-1:] + got[:-1]
    if (shape[order[0]] if order else 1) != (other[lead[0]] if lead else 1):
        return None
    return Layout(lead)


def PhysicalMinorToMajor(layout):
    """Upstream's `minor_to_major` for one of these layouts: the bank's order
    minor first, and the lane axis MOST MAJOR -- the bank is the outermost thing
    an address picks. Empty for one bank, which has no lane axis to place."""
    order = layout.minor_to_major() if layout is not None else []
    return order[1:] + order[:1]
