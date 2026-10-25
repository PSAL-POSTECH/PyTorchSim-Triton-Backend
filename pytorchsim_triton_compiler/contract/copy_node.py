"""The resident mm, as v6e XLA runs a GEMM: one operand copied whole onto the chip first (a copy node, its
bytes inside the kernel's ELF .spad), the other streamed by output tiles, K whole. The front end builds its
kernel by `plan`; the compiler budgets its ELF by `budget`."""
from typing import NamedTuple, Optional


class Plan(NamedTuple):
    """`resident` 0 (A) or 1 (B) held whole, the other streamed by `tile` rows (A) or columns (B); None, and
    `why`."""
    resident: Optional[int]
    tile: Optional[int]
    why: str


def plan(M, N, K, in_bytes, acc_bytes, out_bytes, lanes, spad_lane_bytes, max_numel, array_width):
    """XLA's order: the larger operand (B on a tie) held if it fits the whole scratchpad beside the other's
    smallest tile, else the smaller; then the tile best_tile picks beside it. None for M or N of 1."""
    if M == 1 or N == 1:
        return Plan(None, None, f"M={M}, N={N}: a vector product, which XLA prefetches nothing for")
    tried = []
    for res in order({0: M * K, 1: K * N}):
        tile = best_tile(res, M, N, K, in_bytes, acc_bytes, out_bytes, lanes, spad_lane_bytes, max_numel,
                         array_width)
        if tile is not None:
            return Plan(res, tile, f"{'AB'[res]} ({(M * K if res == 0 else K * N) * in_bytes} B) held, the other "
                                   f"streamed by {tile}")
        tried.append(f"{'AB'[res]} does not fit beside a {lanes}-wide tile")
    return Plan(None, None, "; ".join(tried))


def order(sizes):
    """The operands {index: size} in the order XLA prefers to prefetch one whole: the larger first, on a tie
    the later (B before A). XLA prefetches at most one; a kernel that loads several whole takes the first as
    its copy node."""
    return sorted(sizes, key=lambda r: (-sizes[r], -r))


def best_tile(res, M, N, K, in_bytes, acc_bytes, out_bytes, lanes, spad_lane_bytes, max_numel, array_width):
    """Of the tiles lanes * 2^i up to the streamed extent rounded up with which `fits`, the one the matrix
    unit walks the least for (a tile computed whole, at least `array_width` columns, Target.matmul_width), the
    larger on a tie."""
    extent = N if res == 0 else M
    tiles, t = [], lanes
    while fits(res, t, M, N, K, in_bytes, acc_bytes, out_bytes, lanes, spad_lane_bytes, max_numel):
        tiles.append(t)
        if t >= extent:
            break
        t *= 2
    return min(tiles, key=lambda t: (-(-extent // t) * max(t, array_width), -t), default=None)


def fits(res, t, M, N, K, in_bytes, acc_bytes, out_bytes, lanes, spad_lane_bytes, max_numel):
    """Whether the resident mm's buffers, as the compiler allocates them (`fitted`), fit the whole scratchpad,
    a lane's bank and all of it, each block at most `max_numel`: the held operand, two streamed tiles and
    products, a stored tile."""
    f = lambda v: fitted(v, lanes)
    if res == 0:
        held, streamed, out = (f(M), f(K)), (f(K), t), (f(M), t)
    else:
        held, streamed, out = (f(K), f(N)), (t, f(K)), (t, f(N))
    tiles = [held + (in_bytes,)] + [streamed + (in_bytes,), out + (acc_bytes,)] * 2 + [out + (out_bytes,)]
    lane = sum(r * c * b // min(r, c, lanes) for r, c, b in tiles)
    blocks = [(block(M, lanes), block(K, lanes)), (block(K, lanes), block(N, lanes))][res]
    return (blocks[0] * blocks[1] <= max_numel and all(r * c <= max_numel for r, c, _ in tiles)
            and lane <= spad_lane_bytes and sum(r * c * b for r, c, b in tiles) <= spad_lane_bytes * lanes)


def budget(spad_lane_bytes, spad_buffers, holds):
    """A lane's bytes an ELF may claim: all of the bank for a kernel that holds an operand (TOGSim gives it
    its core alone, so nothing else is in flight), else the bank over the copies of each buffer the compiler
    keeps."""
    return spad_lane_bytes if holds else spad_lane_bytes // spad_buffers


def fitted(v, lanes):
    """The extent the compiler allocates for `v` (pytorchsim-fit-to-numel): `v` itself where it is a whole
    number of lanes past them, else its Triton block."""
    return v if v > lanes and v % lanes == 0 else block(v, lanes)


def block(v, lanes):
    """`v` as a Triton block extent: the next power of two, a multiple of `lanes` past them."""
    p = 1 << (int(v) - 1).bit_length()
    return p if p <= lanes else -(-p // lanes) * lanes
