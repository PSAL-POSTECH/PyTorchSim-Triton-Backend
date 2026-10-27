"""The resident mm: one operand copied whole onto the chip first (a copy node, its bytes inside the kernel's
ELF .spad), the other streamed by output tiles, K whole. The front end builds its kernel by `plan`; the
compiler budgets its ELF by `budget`."""
import math
from typing import NamedTuple, Optional


def graph_prefetches(subtarget):
    """How many arguments a graph (its kernels) may hold resident on `subtarget`, Subtarget.resident_operands
    (compiler.resident_operands): the most one KernelSpec.graph_prefetch names."""
    return subtarget.resident_operands


class Plan(NamedTuple):
    """`resident` 0 (A) or 1 (B) held whole, the other streamed by `tile` rows (A) or columns (B); None, and
    `why`."""
    resident: Optional[int]
    tile: Optional[int]
    why: str


def plan(M, N, K, in_bytes, acc_bytes, out_bytes, subtarget, max_numel):
    """The operand `order` puts first (the larger, A on a tie) held if it fits `subtarget` (fits) beside the
    other's smallest tile, else the other; then the tile best_tile picks beside it. None for M or N of 1."""
    if M == 1 or N == 1:
        return Plan(None, None, f"M={M}, N={N}: a vector product, which holds no operand")
    tried = []
    for res in order({0: M * K, 1: K * N}):
        tile = best_tile(res, M, N, K, in_bytes, acc_bytes, out_bytes, subtarget, max_numel)
        if tile is not None:
            return Plan(res, tile, f"{'AB'[res]} ({(M * K if res == 0 else K * N) * in_bytes} B) held, the other "
                                   f"streamed by {tile}")
        tried.append(f"{'AB'[res]} does not fit beside a {subtarget.vector_lanes}-wide tile")
    return Plan(None, None, "; ".join(tried))


class Candidate(NamedTuple):
    """A tensor a graph may hold, as prefetch_order ranks it: its `bytes`, the elements its uses produce
    together, the position of its latest use and its buffer's id."""
    bytes: int
    use_elements: int
    latest_use: int
    buffer_id: int


def prefetch_order(candidates):
    """The {key: Candidate} keys in the order the graph holds them whole: the larger first, then the more
    elements its uses produce, the earlier latest use, the lower buffer id."""
    return sorted(candidates, key=lambda k: (-candidates[k].bytes, -candidates[k].use_elements,
                                             candidates[k].latest_use, candidates[k].buffer_id))


def order(sizes):
    """A GEMM's operands {index: size} by prefetch_order: one use each, the same one, and the index (A 0, B 1)
    their buffer ids as for parameters, so the larger first and A on a tie."""
    return prefetch_order({r: Candidate(s, 0, 0, r) for r, s in sizes.items()})


def best_tile(res, M, N, K, in_bytes, acc_bytes, out_bytes, subtarget, max_numel):
    """Of the tiles lanes * 2^i up to the streamed extent rounded up with which `fits`, the one the matrix
    unit walks the least for (a tile computed whole, at least Subtarget.matmul_width columns), the larger on a
    tie."""
    extent, array_width = (N if res == 0 else M), subtarget.matmul_width
    tiles, t = [], subtarget.vector_lanes
    while fits(res, t, M, N, K, in_bytes, acc_bytes, out_bytes, subtarget, max_numel):
        tiles.append(t)
        if t >= extent:
            break
        t *= 2
    return min(tiles, key=lambda t: (-(-extent // t) * max(t, array_width), -t), default=None)


def fits(res, t, M, N, K, in_bytes, acc_bytes, out_bytes, subtarget, max_numel):
    """Whether the resident mm's buffers, as the compiler allocates them (`fitted`), fit `subtarget`'s
    scratchpad, a lane's bank and all of it, each block at most `max_numel`: the held operand, which
    resident_fits holds, two streamed tiles and products, a stored tile."""
    lanes, spad_lane_bytes = subtarget.vector_lanes, subtarget.spad_size
    f = lambda v: fitted(v, lanes)
    if res == 0:
        held, streamed, out = (f(M), f(K)), (f(K), t), (f(M), t)
    else:
        held, streamed, out = (f(K), f(N)), (t, f(K)), (t, f(N))
    tiles = [held + (in_bytes,)] + [streamed + (in_bytes,), out + (acc_bytes,)] * 2 + [out + (out_bytes,)]
    lane = sum(tile_lane_bytes((r, c), b, lanes) for r, c, b in tiles)
    blocks = [(block(M, lanes), block(K, lanes)), (block(K, lanes), block(N, lanes))][res]
    return (resident_fits(subtarget, (M, K) if res == 0 else (K, N), in_bytes)
            and blocks[0] * blocks[1] <= max_numel and all(r * c <= max_numel for r, c, _ in tiles)
            and lane <= spad_lane_bytes and sum(r * c * b for r, c, b in tiles) <= spad_lane_bytes * lanes)


def resident_fits(subtarget, shape, elem_bytes):
    """Whether a tensor of `shape` and `elem_bytes` may be held whole on `subtarget`: its graph prefetches one
    and the tensor, as the compiler allocates it (resident_lane_bytes), fits resident_budget. What the
    front end elects by and plan-copy-node holds by, from the tensor alone; the kernel's own buffers beside
    it are the ELF's .spad check's (budget)."""
    return graph_prefetches(subtarget) > 0 and \
        resident_lane_bytes(subtarget, shape, elem_bytes) <= resident_budget(subtarget)


def resident_lane_bytes(subtarget, shape, elem_bytes):
    """A lane's bytes of a tensor of `shape` held whole: each axis but a unit one `fitted`, banked across the
    smallest (tile_lane_bytes)."""
    dims = [fitted(d, subtarget.vector_lanes) for d in shape if d > 1] or [1]
    return tile_lane_bytes(dims, elem_bytes, subtarget.vector_lanes)


def resident_budget(subtarget):
    """A lane's bytes a resident operand may take on `subtarget`: Subtarget.resident_lane_bytes, the bank but
    the share the graph keeps for its kernels' own buffers."""
    return subtarget.resident_lane_bytes


def tile_lane_bytes(dims, elem_bytes, lanes):
    """A lane's bytes of a tile of `dims`, banked across its smallest axis, at most `lanes` of it."""
    return math.prod(dims) * elem_bytes // min(min(dims), lanes)


def budget(spad_lane_bytes, spad_buffers, holds):
    """A lane's bytes an ELF may claim: all of the bank for a kernel that holds an operand (TOGSim gives it
    its core alone, so nothing else is in flight), else the bank over the copies of each buffer the compiler
    keeps."""
    return spad_lane_bytes if holds else spad_lane_bytes // spad_buffers


def fitted(v, lanes):
    """The extent the compiler allocates for `v` (pytorchsim-drop-unread-padding): `v` itself where it is a
    whole number of lanes past them, else its Triton block."""
    return v if v > lanes and v % lanes == 0 else block(v, lanes)


def block(v, lanes):
    """`v` as a Triton block extent: the next power of two, a multiple of `lanes` past them."""
    p = 1 << (int(v) - 1).bit_length()
    return p if p <= lanes else -(-p // lanes) * lanes
