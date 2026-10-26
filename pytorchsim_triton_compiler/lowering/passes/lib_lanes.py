"""lib_lanes: how a tile sits on the machine's lanes -- the rows each lane holds, an axis longer than the
lanes folded onto them -- as arithmetic over shapes and (lane, row) pairs, with no IR."""


def need_per_lane(numel, lane_extent, vectorlane):
    """Elements one lane actually HOLDS of a `numel`-element tile: its rows, once per lane-full of
    a lane axis longer than the machine -- a last, partial lane-full as deep as a whole one, since
    the lanes it leaves idle hold nothing else."""
    if lane_extent and lane_extent > vectorlane:
        return lane_fulls(lane_extent, vectorlane) * (numel // lane_extent)
    used = lane_extent or vectorlane
    return -(-numel // max(used, 1))


def lane_fulls(extent, vectorlane):
    """How many times a lane axis of `extent` wraps the machine's lanes -- the last time partially,
    where `extent` is no multiple of them."""
    return -(-extent // vectorlane)


def fold_onto_lanes(pairs, s_shape, d_shape, lanes):
    """`pairs` on the lanes the machine HAS: lane-full t of an axis 0 longer than `lanes` sits t
    times the other axes' rows deeper, as the DMA stages it and need_per_lane reserves it; a last,
    partial lane-full (a scan pads to one and a half) on the lanes it reaches, as deep as a whole."""
    def fold(shape):
        if not shape or shape[0] <= lanes:
            return lambda at: at
        rows = 1
        for d in shape:
            rows *= d
        rows //= shape[0]
        return lambda at: (at[0] % lanes, at[0] // lanes * rows + at[1])
    if pairs is None:
        return None
    fs, fd = fold(s_shape), fold(d_shape)
    return [(fs(f), fd(t)) for f, t in pairs]
