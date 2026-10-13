"""torchsim-lower-matmul: every torchsim.dot onto the systolic array, its subtiles streamed (only
those a masked A or B load leaves live); each pop added into C, written at the first K into a C
proven zero; a loop only reading C after the dot runs at the last K's pops, or is marked why not."""

import functools
import sys
from collections import namedtuple

from ... import toolchain
from ...contract import vcix_format

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from .lib_bank_layout import (BANK_VIEW_OP, bank_layout, bank_view, index_bank_casts,  # noqa: E402
                              lanes_of)
from ...dialects.torchsim import TransferOp, WaitOp, is_load  # noqa: E402
from .lib_ir import (DOT_OP, VIEW_OPS, array_kind, build_trip_loop, const_index,  # noqa: E402
                     elem_bits, loop_bounds, op_of, producer, transfer_read, transfer_write,
                     writes_through_this_operand, zero_of)
from .lib_indexing_maps import MATMUL_K, MATMUL_M, MATMUL_N, matmul_axis  # noqa: E402
from .lib_vcix import bitcast, i64 as a64, vcix, word_vector  # noqa: E402

MATMUL_OPS = (DOT_OP,)

FORBIDS = MATMUL_OPS

MARKERS = MATMUL_OPS

_Gemm = namedtuple(
    "_Gemm",
    "spad_map a64 rvl rvl_acc elen nr_element nr_acc M_val push_val M_LOOP arith reach epi zero")

_GemmOperand = namedtuple("_GemmOperand", "value vec word pad")

class ArrayRefusal(Exception):
    """The array cannot take this matmul."""


def run(module, *, vectorlane, vlen, queue_entries, **_):
    """Every matmul in `module` lowered in place; `queue_entries` is the array's input and output
    queue size."""
    from .lib_ir import walk
    index_bank_casts(module)
    dots = [o for o in walk(module.operation) if o.name in MATMUL_OPS]
    inits = [_zero_init(o, vectorlane, vlen, queue_entries) for o in dots]
    dead = _dead_initializers([i for i in inits if i is not None])
    for o, init in zip(dots, inits):
        _rewrite_matmul(o, vectorlane, vlen, queue_entries, init is not None)
    for nest in dead:
        nest.erase()
    return module


def _rewrite_matmul(op, SS, vlen, queue_entries, zero=False):
    from mlir.dialects import arith

    A, B, C = op.operands[0], op.operands[1], op.operands[2]
    mtA, mtB, mtC = (ir.MemRefType(A.type), ir.MemRefType(B.type),
                     ir.MemRefType(C.type))
    elt, acc_elt = mtA.element_type, mtC.element_type
    M, N, K = _get_mnk(op)
    sysM, sysN, sysK = _get_systolic_tile(M, N, K, SS, op)
    _refuse_unlowerable(op, elt, acc_elt, mtB.element_type)
    elen = elem_bits(elt)
    nr_element = vlen // elen
    nr_acc = vlen // elem_bits(acc_elt)
    i64 = ir.IntegerType.get_signless(64)

    vectorType, vectorMType, vectorCType = _gemm_vector_types(
        M, elt, acc_elt, nr_element, nr_acc)
    wordType, wordMType = word_vector(vectorType), word_vector(vectorMType)
    wordCType = word_vector(vectorCType)
    bank_map = _get_spad_map()

    push_length = min(M, sysM)
    M_LOOP = min(M, push_length)

    ip = ir.InsertionPoint(op)
    with ip:
        c0 = const_index(0)
        rvl = arith.ConstantOp(i64, a64(nr_element)).result
        rvl_acc = (rvl if nr_acc == nr_element
                   else arith.ConstantOp(i64, a64(nr_acc)).result)
        K_val, M_val = const_index(K), const_index(M)
        push_val = const_index(push_length)
        bankA, bankB, bankC = (bank_view(x, SS) for x in (A, B, C))
        zero_pad = zero_of(elt)
        acc_pad = zero_pad if acc_elt == elt else zero_of(acc_elt)

    grid = (N // sysN, K // sysK, M // push_length)
    g = _Gemm(bank_map, a64, rvl, rvl_acc, elen, nr_element, nr_acc,
              M_val, push_val, M_LOOP, False, (grid, K, M, push_length, SS), None, zero)
    b = _GemmOperand(bankB, vectorType, wordType, zero_pad)
    a = _GemmOperand(bankA, vectorMType, wordMType, zero_pad)
    c = _GemmOperand(bankC, vectorCType, wordCType, acc_pad)
    depth = _get_stream_depth(M_LOOP, nr_element, nr_acc, queue_entries,
                              grid[0] * grid[1] * grid[2])

    if depth > 1:
        padding = (None, None, None)
        if K <= sysK:
            padding = _build_short_k_padding(ip, K, SS, elen, wordType, wordMType)
        live = _live_subtiles(op, grid, (sysN, sysK, push_length), depth)
        if set(live) <= {1}:
            g = g._replace(epi=_find_epilogue(op, g, c, ip))
        if live:
            _build_live_stream(ip, g._replace(arith=True), a, b, c, grid, live, depth,
                               K, K_val, SS, padding)
        else:
            _build_stream(ip, g, a, b, c, grid, depth, K, K_val, SS, padding)
        if g.epi is not None:
            g.epi.nest.erase()
        op.erase()
        return

    g = g._replace(epi=_find_epilogue(op, g, c, ip))
    if g.epi is not None:
        _build_waits(ip, g.epi)
    body_ip = ip
    n_idx = c0
    k_idx = c0
    if N > sysN:
        n_idx, body_ip = build_trip_loop(body_ip, N // sysN)
    zero_vector = None
    a_zero = live_lane = None
    if K > sysK:
        k_idx, body_ip = build_trip_loop(body_ip, K // sysK)
    else:
        zero_vector, a_zero, live_lane = _build_short_k_padding(
            body_ip, K, SS, elen, wordType, wordMType)

    with body_ip:
        _build_b_pushes(g, b, n_idx, k_idx, K, K_val, SS, zero_vector)

    m_idx = c0
    if M > push_length:
        m_idx, body_ip = build_trip_loop(body_ip, M // push_length)

    with body_ip:
        _build_a_pushes(g, a, k_idx, m_idx, live_lane, a_zero)
        _build_compute(g)
        last = True if K <= sysK or g.epi is None else _last_k(grid, k_idx)
        first = zero and (K <= sysK or _first_k(k_idx))
        _build_c_pops(g, c, n_idx, m_idx, last, first)
    if g.epi is not None:
        g.epi.nest.erase()
    op.erase()


def _get_stream_depth(m_loop, nr_element, nr_acc, queue_entries, subtiles):
    """How many subtiles are pushed ahead of their pops: as many as the input and output queues
    hold together, so no push waits on a pop behind it; 1 when a subtile pops other rows."""
    pushed, popped = _rows(m_loop, nr_element), _rows(m_loop, nr_acc)
    if pushed != popped:
        return 1
    return max(1, min(subtiles, 2 * queue_entries // pushed))


def _rows(extent, per_insn):
    return -(-extent // per_insn) * per_insn


def _build_stream(ip, g, a, b, c, grid, depth, K, K_val, SS, padding):
    """The `grid` (n, kk, m) subtiles as one stream: subtile t's rows go in between the pops of
    t - depth. The first `depth` subtiles and the last pops are unrolled; the rest loops, cut where
    a zero C's n = 0 first-K pops end and where an epilogue's last-K pops begin, its waits there."""
    total = grid[0] * grid[1] * grid[2]
    with ip:
        for t in range(depth):
            _build_subtile(g, a, b, c, grid, t, None, K, K_val, SS, padding)
    end = max(total - depth, 0)
    cut = min((grid[1] - 1) * grid[2] if g.epi else total, end)
    edges = sorted({0, cut, end, min(grid[2], end) if g.zero else 0})
    waited = g.epi is None
    for lo, hi in zip(edges, edges[1:]):
        if lo >= cut and not waited:
            _build_waits(ip, g.epi)
            waited = True
        last = False if hi <= cut else (True if grid[0] == 1 else None)
        iv, body = _build_range_loop(ip, lo, hi)
        with body:
            t = _build_apply(ir.AffineMap.get(1, 0, [ir.AffineDimExpr.get(0) + depth]), [iv])
            _build_subtile(g, a, b, c, grid, t, iv, K, K_val, SS, padding, last,
                           _span_first(g, grid, lo, hi))
    if not waited:
        _build_waits(ip, g.epi)
    with ip:
        for t in range(total - depth, total):
            n, _, m = _coords(grid, t)
            last = t // grid[2] % grid[1] == grid[1] - 1
            first = g.zero and t // grid[2] % grid[1] == 0
            for pop in _c_pop_chunks(g, c, n, m, last, first):
                pop()


def _span_first(g, grid, lo, hi):
    """Whether the subtiles a loop drains over [lo, hi) are first-K ones: a bool when all of them
    agree, else None, left to each subtile's kk."""
    seen = {d // grid[2] % grid[1] == 0 for d in range(lo, hi)} if g.zero else {False}
    return seen.pop() if len(seen) == 1 else None


def _first_k(kk):
    from mlir.dialects import arith
    return arith.CmpIOp(arith.CmpIPredicate.eq, kk, const_index(0)).result


def _build_range_loop(ip, lo, hi):
    if lo == 0:
        return build_trip_loop(ip, hi)
    from mlir.dialects import affine
    with ip:
        lp = affine.AffineForOp(lo, hi, 1)
    with ir.InsertionPoint(lp.body):
        affine.AffineYieldOp([])
    return lp.induction_variable, ir.InsertionPoint.at_block_terminator(lp.body)


def _masked_extents(op):
    """{matmul dim: runtime count of its live elements} from A's and B's masked loads.

    A block wider than its matrix (N = 1152 in BLOCK_N 2048) masks the tail; the subtiles over it
    hold nothing the store keeps, yet the stream walks all of them. The count is the load's clamp
    high on the axis that carries the dim. K is trusted only when the masked lanes are filled with
    0: then the skipped subtiles add nothing."""
    out = {}
    for slot in (0, 1):
        buf = op.operands[slot]
        loads = [op_of(u.owner) for u in buf.uses
                 if op_of(u.owner).name == TransferOp.OPERATION_NAME and is_load(op_of(u.owner))]
        if len(loads) != 1:
            continue
        tx = loads[0]
        v = tx.opview
        if (v.masked_axes is None or v.masked_fill is None or v.index is not None
                or v.subtile_size is not None or v.cut is not None or v.dyn_fill is not None
                or v.sram != buf):
            continue
        if tx.operation.block != op.operation.block or not tx.operation.is_before_in_block(
                op.operation):
            continue                      # its clamp would not dominate the stream
        axes = [ir.IntegerAttr(x).value for x in v.masked_axes]
        clamp = list(v.clamp)
        if len(clamp) != 2 * len(axes):
            continue
        zero_fill = ir.IntegerAttr(v.masked_fill).value == 0
        for i, axis in enumerate(axes):
            for dim in (MATMUL_M, MATMUL_N, MATMUL_K):
                try:
                    hit = matmul_axis(op, slot, dim) == axis
                except ValueError:
                    hit = False
                if hit and (dim != MATMUL_K or zero_fill) and dim not in out:
                    out[dim] = clamp[2 * i + 1]
    return out


def _live_subtiles(op, grid, sub, depth):
    """{grid axis: (live elements, subtile size, least count)} for the axes of the (N, K, M) grid a
    masked load lets the stream cut, or {} when it can cut none. The least counts keep the live
    subtiles at `depth` or more, so the stream's unrolled head and tail are always whole: an axis
    that would need its full count to hold that is left as it is."""
    extents = _masked_extents(op)
    live = {i: extents[d] for i, d in enumerate((MATMUL_N, MATMUL_K, MATMUL_M))
            if d in extents and grid[i] > 1}
    least = {i: 1 for i in live}
    for i in live:
        others = 1
        for j in range(3):
            if j != i:
                others *= least.get(j, grid[j])
        least[i] = min(grid[i], -(-depth // others))
    return {i: (live[i], sub[i], least[i]) for i in live if least[i] < grid[i]}


def _build_live_stream(ip, g, a, b, c, grid, live, depth, K, K_val, SS, padding):
    """_build_stream over the subtiles a mask leaves live: the same head, loop and tail, with the
    count of each cut axis a runtime value -- ceil(live / subtile), held to [least, its grid
    count]. The first subtile is always (0, 0, 0), so the counts are worked out after it."""
    from mlir.dialects import arith, scf
    with ip:
        _build_subtile(g, a, b, c, grid, 0, None, K, K_val, SS, padding)
        counts = list(grid)
        for i, (extent, sub, least) in live.items():
            n = arith.DivSIOp(arith.AddIOp(extent, const_index(sub - 1)).result,
                              const_index(sub)).result
            n = arith.MinSIOp(n, const_index(grid[i])).result
            counts[i] = arith.MaxSIOp(n, const_index(least)).result
        counts = tuple(counts)
        total = _mul(_mul(counts[0], counts[1]), counts[2])
        for t in range(1, depth):
            _build_subtile(g, a, b, c, counts, const_index(t), None, K, K_val, SS, padding)
        rest = arith.SubIOp(total, const_index(depth)).result
        cut = rest
        if g.epi is not None:
            first = _mul(arith.SubIOp(_ix(counts[1]), const_index(1)).result, counts[2])
            cut = arith.MinSIOp(_ix(first), rest).result
    spans = [(_const_at(ip, 0), cut, False)]
    if g.epi is not None:
        _build_waits(ip, g.epi)
        spans.append((cut, rest, True if grid[0] == 1 else None))
    for lo, hi, last in spans:
        with ip:
            loop = scf.ForOp(lo, hi, const_index(1))
        with ir.InsertionPoint(loop.body):
            iv = loop.induction_variable
            t = arith.AddIOp(iv, const_index(depth)).result
            _build_subtile(g, a, b, c, counts, t, iv, K, K_val, SS, padding, last,
                           None if g.zero else False)
            scf.YieldOp([])
    with ip:
        for back in range(depth, 0, -1):
            t = arith.SubIOp(total, const_index(back)).result
            n, kk, m = _coords(counts, t, True)
            last = _last_k(counts, kk) if g.epi is not None else False
            first = _first_of(g, counts, kk)
            for pop in _c_pop_chunks(g, c, n, m, last, first):
                pop()


def _const_at(ip, v):
    with ip:
        return const_index(v)


def _last_k(grid, kk):
    from mlir.dialects import arith
    top = (const_index(grid[1] - 1) if isinstance(grid[1], int)
           else arith.SubIOp(grid[1], const_index(1)).result)
    return arith.CmpIOp(arith.CmpIPredicate.eq, kk, top).result


def _ix(v):
    return const_index(v) if isinstance(v, int) else v


def _mul(x, y):
    from mlir.dialects import arith
    if isinstance(x, int) and isinstance(y, int):
        return x * y
    if x == 1:
        return y
    if y == 1:
        return x
    return arith.MulIOp(_ix(x), _ix(y)).result


def _first_of(g, grid, kk):
    """Whether subtile kk is a first-K one whose pops write C: a bool, or an i1 on kk."""
    if not g.zero:
        return False
    if isinstance(grid[1], int) and grid[1] == 1:
        return True
    return _first_k(kk)


def _build_subtile(g, a, b, c, grid, t, drained, K, K_val, SS, padding, last=False,
                   first=False):
    """Subtile t: its weights if they change here, its pushes after each pop of subtile
    `drained` (an index, or None) that keeps the pops ahead, then the rest of those pops; `last`
    and `first` say whether `drained` is a last-/first-K subtile: True, False, or None (test kk)."""
    from mlir.dialects import arith, scf
    zero_vector, a_zero, live_lane = padding
    n, kk, m = _coords(grid, t, g.arith)
    if isinstance(t, int):
        if t % grid[2] == 0:
            _build_b_pushes(g, b, n, kk, K, K_val, SS, zero_vector)
    elif isinstance(grid[2], int) and grid[2] == 1:
        _build_b_pushes(g, b, n, kk, K, K_val, SS, zero_vector)
    else:
        weights = scf.IfOp(arith.CmpIOp(arith.CmpIPredicate.eq, m, const_index(0)).result)
        with ir.InsertionPoint(weights.then_block):
            _build_b_pushes(g, b, n, kk, K, K_val, SS, zero_vector)
            scf.YieldOp([])
    drain = []
    if drained is not None:
        dn, dkk, dm = _coords(grid, drained, g.arith)
        if last is None:
            last = _last_k(grid, dkk) if g.epi is not None else False
        if first is None:
            first = _first_of(g, grid, dkk)
        drain = _c_pop_chunks(g, c, dn, dm, last, first)
    pushed = popped = 0
    for rows, push in _a_push_chunks(g, a, kk, m, live_lane, a_zero):
        while drain and popped < pushed + rows:
            popped += drain.pop(0)()
        push()
        pushed += rows
    _build_compute(g)
    for pop in drain:
        pop()


def _coords(grid, t, arith_form=False):
    """(n, kk, m) of subtile t of an (N, K, M) grid, m innermost: constants for a constant t, an
    affine map of the stream's counter, or -- `arith_form`, the live stream, whose counts and
    counter no affine map can take -- index arithmetic."""
    _, nk, nm = grid
    if isinstance(t, int):
        return tuple(const_index(v) for v in (t // (nk * nm), t // nm % nk, t % nm))
    if arith_form:
        from mlir.dialects import arith
        across = arith.DivSIOp(t, _ix(nm)).result
        return (arith.DivSIOp(t, _ix(_mul(nk, nm))).result,
                arith.RemSIOp(across, _ix(nk)).result,
                arith.RemSIOp(t, _ix(nm)).result)
    d0 = ir.AffineDimExpr.get(0)
    k = ir.AffineConstantExpr.get

    def div(e, v):
        return ir.AffineFloorDivExpr.get(e, k(v))

    def mod(e, v):
        return ir.AffineModExpr.get(e, k(v))
    exprs = (div(d0, nk * nm), mod(div(d0, nm), nk), mod(d0, nm))
    return tuple(_build_apply(ir.AffineMap.get(1, 0, [e]), [t]) for e in exprs)


def _refuse_unlowerable(op, elt, acc_elt, b_elt):
    if array_kind(op) == "batch_matmul":
        raise ArrayRefusal(f"a batched torchsim.dot: this lowering walks n, k and m and has no batch "
                           f"loop, so it would compute the first batch only: {op}")
    if b_elt != elt:
        raise NotImplementedError(
            f"matmul operands A ({elt}) and B ({b_elt}) have "
            f"different element types; every push is typed from A's element type at one sew, "
            f"so B's elements would be read as A's")
    if isinstance(acc_elt, ir.BF16Type):
        raise NotImplementedError(
            f"a bf16 accumulator ({acc_elt}): the pop is added into C, and bf16 arithmetic "
            f"has no instruction on this machine; accumulate in f32")
    for t in (elt, acc_elt):
        if isinstance(t, ir.IntegerType):
            raise NotImplementedError(
                f"integer matmul ({t}) has no path to the systolic array: spike's "
                f"array is float throughout (systolic_array.h -- serialisers, "
                f"weights and the accumulator are all `float`), so there is no "
                f"integer MAC to lower to")
    _refuse_accumulator_in_its_loop(op)


def _refuse_accumulator_in_its_loop(op):
    innermost = None
    parent = op.operation.parent
    while parent is not None:
        if parent.name in ("affine.for", "scf.for") and innermost is None:
            innermost = parent
        parent = parent.parent

    if innermost is not None:
        owner = op.operands[2].owner
        owner = op_of(owner)
        anc = getattr(owner, "parent", None)
        while anc is not None:
            if anc == innermost:
                raise NotImplementedError(
                    "the matmul accumulator is allocated inside its own loop -- "
                    "that loop is not a reduction axis")
            anc = anc.parent


def _get_systolic_tile(m, n, k, ss, what):
    for d, name in ((m, "M"), (n, "N"), (k, "K")):
        if d > ss and d % ss:
            raise ArrayRefusal(
                f"matmul {name}={d} must be a multiple of the systolic size "
                f"{ss} when larger than it: {what}")
    return [min(m, ss), min(n, ss), min(k, ss)]


def _get_mnk(op):
    mtA, mtB = ir.MemRefType(op.operands[0].type), ir.MemRefType(op.operands[1].type)
    return (mtA.shape[matmul_axis(op, 0, MATMUL_M)],
            mtB.shape[matmul_axis(op, 1, MATMUL_N)],
            mtA.shape[matmul_axis(op, 0, MATMUL_K)])


def _gemm_vector_types(M, elt, acc_elt, nr_element, nr_acc):
    return (ir.VectorType.get([nr_element], elt),
            ir.VectorType.get([max(min(M, nr_element), 2)], elt),
            ir.VectorType.get([max(min(M, nr_acc), 2)], acc_elt))


def _build_short_k_padding(body_ip, K, SS, elen, word_ty, word_m_ty):
    from mlir.dialects import arith, vector

    def _zero_word(ty):
        return arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(
            ty, ir.IntegerAttr.get(
                ir.IntegerType.get_signless(elen), 0))).result

    with body_ip:
        zero_vector = _zero_word(word_ty)
        if K >= SS:
            return zero_vector, None, None
        a_zero = _zero_word(word_m_ty)
        idx_ty = ir.VectorType.get([word_m_ty.shape[0]], ir.IndexType.get())
        lane = ir.Operation.create(
            "torchsim.vlane_idx", results=[idx_ty], operands=[],
            attributes={}).results[0]
        live_lane = arith.CmpIOp(
            arith.CmpIPredicate.ult, lane,
            vector.BroadcastOp(idx_ty, const_index(K)).result).result
    return zero_vector, a_zero, live_lane


def _push_format(vec_ty):
    """The rs1 code a push of `vec_ty` carries, as contract/vcix_format names it."""
    if isinstance(ir.VectorType(vec_ty).element_type, ir.BF16Type):
        return vcix_format.BF16
    return vcix_format.NATIVE


def _build_b_pushes(g, b, n_idx, k_idx, K, K_val, SS, zero_vector):
    for i in range(0, SS, g.nr_element):
        if i < K:
            at = _build_spad_at(g, n_idx, k_idx, i, K_val, const_index(SS))
            fits = _in_bank(b, g.reach[0][0], K, g.reach[0][1], SS, i)
            wv = bitcast(transfer_read(b.vec, b.value, at, b.pad, fits), b.word)
        else:
            wv = zero_vector
        vcix("vcix.iv", [wv, g.rvl], [],
              {"opcode": g.a64(1), "imm": g.a64(_push_format(b.vec)), "rd": g.a64(0)})


def _build_a_pushes(g, a, k_idx, m_idx, live_lane, a_zero):
    for _, push in _a_push_chunks(g, a, k_idx, m_idx, live_lane, a_zero):
        push()


def _a_push_chunks(g, a, k_idx, m_idx, live_lane, a_zero):
    """Each input push of one subtile, as (rows, emitter), in order."""
    from mlir.dialects import arith

    def push(i):
        at = _build_spad_at(g, k_idx, m_idx, i, g.M_val, g.push_val)
        grid, _, M, rows, _ = g.reach
        fits = _in_bank(a, grid[1], M, grid[2], rows, i)
        iv = bitcast(transfer_read(a.vec, a.value, at, a.pad, fits), a.word)
        if live_lane is not None:
            iv = arith.SelectOp(live_lane, iv, a_zero).result
        vcix("vcix.iv", [iv, g.rvl], [],
              {"opcode": g.a64(0), "imm": g.a64(_push_format(a.vec)), "rd": g.a64(0)})
        return g.nr_element
    return [(g.nr_element, functools.partial(push, i)) for i in range(0, g.M_LOOP, g.nr_element)]


def _build_compute(g):
    vcix("vcix.i", [g.rvl], [],
          {"opcode": g.a64(1), "imm": g.a64(4), "rd": g.a64(0),
           "rs2": g.a64(0), "sew": g.a64(g.elen), "lmul": g.a64(0)})


_ZeroInit = namedtuple("_ZeroInit", "dot acc runs whole copy fill")


def _zero_init(op, SS, vlen, queue_entries):
    """How `op`'s C holds zero over every element its pops touch, or None: its last writer in the
    block is a zero-splat store nest over them, or a nest copying them from a buffer whose own
    last writer is one. `whole` says the pops reach all of them at run time."""
    C = op.operands[2]
    if any(_root(v) == _root(C) for v in op.operands[:2]) or not _plain(C):
        return None
    try:
        M, N, K = _get_mnk(op)
        sysM, sysN, sysK = _get_systolic_tile(M, N, K, SS, op)
    except (ArrayRefusal, ValueError):
        return None
    rows = min(M, sysM)
    nr_element = vlen // elem_bits(ir.MemRefType(op.operands[0].type).element_type)
    nr_acc = vlen // elem_bits(ir.MemRefType(C.type).element_type)
    grid = (N // sysN, K // sysK, M // rows)
    runs = _runs([n * M + m * rows + i for n in range(grid[0]) for m in range(grid[2])
                  for i in range(0, rows, nr_acc)], max(min(M, nr_acc), 2))
    if runs is None:
        return None
    depth = _get_stream_depth(rows, nr_element, nr_acc, queue_entries,
                              grid[0] * grid[1] * grid[2])
    live = _live_subtiles(op, grid, (sysN, sysK, rows), depth) if depth > 1 else {}
    writer = _last_writer(op.operation, C)
    if writer is None:
        return None
    filled = _store_nest(writer, C, None)
    if filled is not None and _covers(filled, runs):
        return _ZeroInit(op.operation, C, runs, set(live) <= {1}, None, (writer, filled))
    copied = _store_nest(writer, C, True)
    if copied is None or not _covers(copied[1], runs):
        return None
    Z = copied[0]
    source = _last_writer(writer, Z)
    filled = None if source is None else _store_nest(source, Z, None)
    if filled is None or not _covers(filled, copied[1]):
        return None
    return _ZeroInit(op.operation, C, runs, set(live) <= {1}, (writer, copied[1]),
                     (source, filled))


def _dead_initializers(inits):
    """The copy nests into a zero-proven C whose pops overwrite all they wrote, nothing using C
    in between, and the zero fills nothing reads any more before a first-K dot overwrites them."""
    copies = [i.copy[0] for i in inits if i.copy is not None and i.whole
              and i.copy[1] == i.runs and _used_between(i.acc, i.copy[0], i.dot) is None]
    dead, fills = list(copies), []
    for i in inits:
        if not any(i.fill[0] == f for f, _ in fills):
            fills.append(i.fill)
    for fill, runs in fills:
        users = _block_users(_access_buffer(_nest(fill)[1][0]), fill.block)
        for o in _ops_after(fill):
            if not _among(o, users) or _among(o, copies):
                continue
            if any(o == i.dot and i.copy is None and i.fill[0] == fill and i.whole
                   and _covers(i.runs, runs) for i in inits):
                dead.append(fill)
            break
    return dead


def _root(v):
    """The buffer `v` names: a global by its symbol, else the value under its view ops."""
    while True:
        p = producer(v)
        if p is not None and p.name == "memref.get_global":
            return ("global", ir.FlatSymbolRefAttr(p.attributes["name"]).value)
        if p is None or not _is_view(p):
            return v
        v = p.operands[0]


def _is_view(o):
    return o.name in VIEW_OPS or o.name == BANK_VIEW_OP


def _plain(v):
    p = producer(v)
    return isinstance(v.type, ir.MemRefType) and (p is None or not _is_view(p))


_Use = namedtuple("_Use", "operand_number")


def _users(v):
    """(op, operand number) of every op that takes `v`'s buffer, past view ops, off the use-lists
    of each value naming that buffer -- for a global, every get_global of it in the function."""
    from .lib_ir import walk
    root = _root(v)
    names = [root]
    if isinstance(root, tuple):
        func = producer(v)
        while func.name != "memref.get_global":
            func = producer(func.operands[0])
        while func.name != "func.func":
            func = func.parent
        names = [o.results[0] for o in walk(func) if o.name == "memref.get_global"
                 and ir.FlatSymbolRefAttr(o.attributes["name"]).value == root[1]]
    out = []
    while names:
        for use in names.pop().uses:
            o = op_of(use.owner)
            if _is_view(o):
                names.extend(o.results)
            else:
                out.append((o, use.operand_number))
    return out


def _block_users(v, block, writes=False):
    """The ops of `block` that hold a user of `v`'s buffer (`writes`: one writing it)."""
    out = []
    for o, k in _users(v):
        if writes and not writes_through_this_operand(o, _Use(k)):
            continue
        while o is not None and o.name != "func.func" and o.block != block:
            o = _parent(o)
        if o is not None and o.name != "func.func" and not _among(o, out):
            out.append(o)
    return out


def _among(o, ops):
    return any(o == x for x in ops)


def _ops_after(op):
    ops = [o.operation for o in op.block.operations]
    return ops[next(i for i, o in enumerate(ops) if o == op) + 1:]


def _last_writer(op, buf):
    """The last op before `op` in its block that writes `buf`'s buffer, or None."""
    ops = [o.operation for o in op.block.operations]
    writers = _block_users(buf, op.block, True)
    for o in reversed(ops[:next(i for i, o in enumerate(ops) if o == op)]):
        if _among(o, writers):
            return o
    return None


def _used_between(buf, a, b, writes=False, first=1):
    """The first op of the block from `a` (`first` = 0) or past it (1) up to, not at, `b` that uses
    (`writes`: writes) `buf`'s buffer, or None."""
    ops = [o.operation for o in a.block.operations]
    lo = next(i for i, o in enumerate(ops) if o == a)
    hi = next(i for i, o in enumerate(ops) if o == b)
    users = _block_users(buf, a.block, writes)
    return next((o for o in ops[lo + first:hi] if _among(o, users)), None)


def _nest(loop):
    """(the perfect affine.for nest from `loop`, the ops of its innermost body)."""
    nest = [loop]
    while True:
        inner = _body_ops(nest[-1])
        if len(inner) == 1 and inner[0].name == "affine.for":
            nest.append(inner[0])
            continue
        return nest, inner


def _store_nest(loop, C, copy):
    """The runs of C's bank a nest at `loop` writes: one zero-splat store (copy None), or
    (source, runs) for one load of another buffer stored unchanged to the same element (copy)."""
    if loop.name != "affine.for":
        return None
    nest, inner = _nest(loop)
    if len(inner) != (2 if copy else 1) or inner[-1].name not in _STORES:
        return None
    store = inner[-1]
    dst, val = _access_buffer(store), store.operands[0]
    if _root(dst) != _root(C) or not _one_vector(val.type) or not _banked_alike(dst, C):
        return None
    if copy:
        load = inner[0]
        if (load.name not in _LOADS or val != load.results[0] or _access(load) != _access(store)
                or _root(_access_buffer(load)) == _root(C)
                or not _banked_alike(_access_buffer(load), C)):
            return None
    elif not _is_zero(_splat_of(val)):
        return None
    bounds = [loop_bounds(lp) for lp in nest]
    if any(b is None or None in b or b[2] <= 0 for b in bounds):
        return None
    ivs = [lp.regions[0].blocks[0].arguments[0] for lp in nest]
    try:
        starts = _consumer_starts(_access(store), bounds, ivs, bank_layout(C)[0])
    except _KeptApart as e:
        loop.attributes[KEPT_APART] = ir.StringAttr.get(f"not read as a zero initializer: {e}")
        return None
    runs = starts and _runs(starts, ir.VectorType(val.type).shape[0])
    if not runs:
        return None
    return (_access_buffer(inner[0]), runs) if copy else runs


def _is_zero(attr):
    if attr is None:
        return False
    for kind in (ir.FloatAttr, ir.IntegerAttr):
        try:
            return kind(attr).value == 0
        except (TypeError, ValueError):
            continue
    return False


def _covers(outer, inner):
    """Is every [lo, hi) run of `inner` inside one of `outer`'s?"""
    return all(any(a <= lo and hi <= b for a, b in outer) for lo, hi in inner)


_Epilogue = namedtuple("_Epilogue", "nest loaded inputs body stored d waits invariant width")

_LOADS = ("affine.vector_load", "vector.load")
_STORES = ("affine.vector_store", "vector.store")
_ELEMENTWISE_DIALECTS = ("arith", "math")
_SPLATS = ("vector.broadcast", "vector.splat")
_MAX_POINTS = 1 << 16

KEPT_APART = "torchsim.kept_apart"


class _KeptApart(Exception):
    """Why a loop beside a dot stays a loop; stamped on it as KEPT_APART."""


def _find_epilogue(op, g, c, ip):
    """The first loop after `op`, past nothing but pure scalar ops, when it reads C's elements only
    to write each, through elementwise ops (by dialect: the bindings' MemoryEffectsOpInterface can
    attach effects, not query them), to the same element of one other buffer."""
    ops = [o.operation for o in op.operation.block.operations]
    at = next(i for i, o in enumerate(ops) if o == op.operation)
    for o in ops[at + 1:]:
        if o.name == "affine.for":
            try:
                return _match_epilogue(op, o, g, c, ip)
            except _KeptApart as e:
                if _among(o, _block_users(op.operands[2], o.block)):
                    o.attributes[KEPT_APART] = ir.StringAttr.get(f"not run at the pops: {e}")
                return None
        if (len(o.regions) or not len(o.results) or o.name.split(".")[0] != "arith"
                or any(isinstance(v.type, (ir.MemRefType, ir.VectorType))
                       for v in list(o.operands) + list(o.results))):
            return None
    return None


def _match_epilogue(op, loop, g, c, ip):
    """_find_epilogue's loop as an _Epilogue, or _KeptApart saying why it stays a loop."""
    C = op.operands[2]
    nest, inner = _nest(loop)
    bounds = [loop_bounds(lp) for lp in nest]
    if any(b is None or None in b or b[2] <= 0 for b in bounds):
        raise _KeptApart("its loop bounds are not constants")
    ivs = [lp.regions[0].blocks[0].arguments[0] for lp in nest]
    loads, stores, waits, body = [], [], [], []
    for o in inner:
        if o.name in _LOADS:
            loads.append(o)
        elif o.name in _STORES:
            stores.append(o)
        elif o.name == WaitOp.OPERATION_NAME:
            waits.append(o)
        elif not len(o.regions) and (o.name.split(".")[0] in _ELEMENTWISE_DIALECTS
                                     or o.name in _SPLATS):
            body.append(o)
        else:
            raise _KeptApart(f"its body holds {o.name}, not an elementwise op")
    if len(stores) != 1:
        raise _KeptApart(f"it stores {len(stores)} times, not once")
    store = stores[0]
    D, stored = _access_buffer(store), store.operands[0]
    reads_c = [x for x in loads if _access_buffer(x) == C]
    if len(reads_c) != 1 or not _one_vector(stored.type):
        raise _KeptApart("it does not read C once into a 1-D vector")
    width_cons = ir.VectorType(stored.type).shape[0]
    access = _access(store)
    buffers = [C, D] + [_access_buffer(x) for x in loads if _access_buffer(x) != C]
    if len({_root(x) for x in buffers}) != len(buffers):
        raise _KeptApart("two of C, its output and its other inputs are one buffer")
    _refuse_overlap(op, loop, D, [x for x in buffers[2:]])
    for x in loads:
        if _access(x) != access or not _one_vector(x.results[0].type) \
                or ir.VectorType(x.results[0].type).shape[0] != width_cons:
            raise _KeptApart("a load does not read the element its store writes")
    if not all(_banked_alike(x, C) for x in buffers[1:]):
        raise _KeptApart("a buffer it touches is not banked as C is")
    defined = {r for o in loads + body for r in o.results}
    for o in body:
        for v in o.operands:
            if v in ivs or (v not in defined and not _dominates(v, op.operation)):
                raise _KeptApart(f"{o.name} takes a value the pops cannot see")
        for v in list(o.operands) + list(o.results):
            if isinstance(v.type, ir.VectorType) and not (
                    _one_vector(v.type) and ir.VectorType(v.type).shape[0] == width_cons):
                raise _KeptApart(f"{o.name} works on a vector of another width")
    if stored not in defined:
        raise _KeptApart("what it stores is not computed in it")
    for w in waits:
        if not all(_dominates(v, op.operation) for v in w.operands):
            raise _KeptApart("a wait in it takes a value the pops cannot see")
    width = ir.VectorType(c.vec).shape[0]
    strides = bank_layout(C)[0]
    cons = _consumer_starts(access, bounds, ivs, strides)
    gr, _, M, rows, SS = g.reach
    pops = [n * M + m * rows + i for n in range(gr[0]) for m in range(gr[2])
            for i in range(0, g.M_LOOP, g.nr_acc)]
    if _runs(cons, width_cons) is None or _runs(cons, width_cons) != _runs(pops, width):
        raise _KeptApart("the elements it reads are not the ones the pops write")
    invariant = {}
    for o in body:
        for v in o.operands:
            if v in defined or v in invariant or not isinstance(v.type, ir.VectorType):
                continue
            splat = _splat_of(v)
            if width != width_cons and splat is None:
                raise _KeptApart("an operand of another width is not a splat")
            invariant[v] = splat
    with ip:
        for v, splat in list(invariant.items()):
            if splat is None:
                invariant[v] = v
                continue
            ty = _retype(v.type, width)
            invariant[v] = _build_constant(ty, splat)
        inputs = []
        for x in loads:
            buf = _access_buffer(x)
            if buf == C:
                continue
            elt = ir.MemRefType(buf.type).element_type
            inputs.append((x.results[0], _GemmOperand(bank_view(buf, SS),
                                                      ir.VectorType.get([width], elt),
                                                      None, zero_of(elt))))
        d = _GemmOperand(bank_view(D, SS),
                         ir.VectorType.get([width], ir.MemRefType(D.type).element_type),
                         None, None)
    return _Epilogue(loop, reads_c[0].results[0], inputs, body, stored, d, waits, invariant,
                     width)


def _refuse_overlap(op, loop, D, inputs):
    """_KeptApart unless, from the dot to the loop, nothing but the loop uses D's buffer -- the
    pops write D from the first last-K one on, while the dot still reads A and B -- and nothing
    writes an input the loop reads."""
    user = _used_between(D, op.operation, loop, first=0)
    if user is not None:
        raise _KeptApart(f"its output buffer is also used by {user.name} before it, which the "
                         f"stream would overwrite while it is still being read")
    for x in inputs:
        user = _used_between(x, op.operation, loop, writes=True, first=0)
        if user is not None:
            raise _KeptApart(f"{user.name} writes an input it reads before it runs")


def _body_ops(loop):
    return [o.operation for o in loop.regions[0].blocks[0].operations
            if o.operation.name != "affine.yield"]


def _access_buffer(o):
    return o.operands[1] if o.name in _STORES else o.operands[0]


def _access(o):
    """(map or None, index operands) of a vector load or store."""
    idx = list(o.operands[2:] if o.name in _STORES else o.operands[1:])
    m = ir.AffineMapAttr(o.attributes["map"]).value if "map" in o.attributes else None
    return (m, tuple(idx))


def _one_vector(ty):
    return isinstance(ty, ir.VectorType) and len(ir.VectorType(ty).shape) == 1


def _banked_alike(x, C):
    """Does element o of `x` sit, in each lane, where element o of C does -- same shape, strides,
    lanes, and neither a view whose own offset would shift it?"""
    if any(producer(v) is not None and producer(v).name in VIEW_OPS for v in (x, C)):
        return False
    tx, tc = ir.MemRefType(x.type), ir.MemRefType(C.type)
    return (list(tx.shape) == list(tc.shape) and bank_layout(x) == bank_layout(C)
            and lanes_of(x) == lanes_of(C) and lanes_of(C) is not None)


def _dominates(v, at):
    p = producer(v)
    anc = at
    if p is None:
        blk = v.owner
        while anc is not None:
            if anc.block == blk:
                return True
            anc = _parent(anc)
        return False
    while anc is not None and anc.block != p.block:
        anc = _parent(anc)
    return anc is not None and p.is_before_in_block(anc)


def _parent(o):
    try:
        return o.parent
    except (ValueError, AttributeError):
        return None


def _consumer_starts(access, bounds, ivs, strides):
    """Each in-bank offset the loop's access starts a vector at, in iteration order; _KeptApart
    when they cannot be listed."""
    import itertools
    trips = 1
    for lo, hi, st in bounds:
        trips *= max(0, -(-(hi - lo) // st))
    if trips > _MAX_POINTS:
        raise _KeptApart(f"its {trips} iterations are more than the {_MAX_POINTS} whose offsets "
                         f"this pass checks one by one")
    m, idx = access
    out = []
    for point in itertools.product(*[range(lo, hi, st) for lo, hi, st in bounds]):
        env = list(zip(ivs, point))
        vals = [_index_value(v, env) for v in idx]
        if None in vals:
            raise _KeptApart("an index is neither a loop counter nor a constant")
        if m is not None:
            dims, syms = vals[:m.n_dims], vals[m.n_dims:]
            vals = [_eval_affine(e, dims, syms) for e in m.results]
        if len(vals) != len(strides):
            raise _KeptApart("its access has not one index per dimension")
        out.append(sum(v * s for v, s in zip(vals, strides)))
    return out


def _index_value(v, env):
    from .lib_ir import const_int
    for iv, x in env:
        if v == iv:
            return x
    return const_int(v)


def _eval_affine(e, dims, syms):
    if isinstance(e, ir.AffineConstantExpr):
        return e.value
    if isinstance(e, ir.AffineDimExpr):
        return dims[e.position]
    if isinstance(e, ir.AffineSymbolExpr):
        return syms[e.position]
    for kind, f in ((ir.AffineAddExpr, lambda x, y: x + y), (ir.AffineMulExpr, lambda x, y: x * y),
                    (ir.AffineModExpr, lambda x, y: x % y),
                    (ir.AffineFloorDivExpr, lambda x, y: x // y),
                    (ir.AffineCeilDivExpr, lambda x, y: -(-x // y))):
        if isinstance(e, kind):
            return f(_eval_affine(e.lhs, dims, syms), _eval_affine(e.rhs, dims, syms))
    raise ValueError(f"an affine expression of no known kind: {e}")


def _runs(starts, width):
    """The [lo, hi) runs `width`-wide vectors at `starts` cover, or None when two overlap."""
    out = []
    for s in sorted(starts):
        if out and s < out[-1][1]:
            return None
        if out and s == out[-1][1]:
            out[-1][1] = s + width
        else:
            out.append([s, s + width])
    return out


def _splat_of(v):
    p = producer(v)
    if p is None or p.name != "arith.constant":
        return None
    attr = ir.DenseElementsAttr(p.attributes["value"])
    return attr.get_splat_value() if attr.is_splat else None


def _retype(ty, width):
    if isinstance(ty, ir.VectorType):
        return ir.VectorType.get([width], ir.VectorType(ty).element_type)
    return ty


def _build_constant(ty, splat):
    from mlir.dialects import arith
    return arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(ty, splat)).result


def _build_waits(ip, epi):
    """The epilogue's waits, once, before its first store."""
    with ip:
        for w in epi.waits:
            WaitOp(w.operands[0], w.operands[1])


def _build_epilogue(epi, total, at, fits):
    """The epilogue on one pop's sum: its other inputs read where the pop's C element sits, its
    ops retyped to the pop's width, the result written there in its own buffer."""
    vals = {epi.loaded: total}
    for loaded, operand in epi.inputs:
        vals[loaded] = transfer_read(operand.vec, operand.value, at, operand.pad, fits)
    for o in epi.body:
        attrs = {n: o.attributes[n] for n in (getattr(x, 'name', x) for x in o.attributes)}
        if o.name == "arith.constant" and isinstance(o.results[0].type, ir.VectorType):
            ty = _retype(o.results[0].type, epi.width)
            attrs["value"] = ir.DenseElementsAttr.get_splat(ty, _splat_of(o.results[0]))
        new = ir.Operation.create(
            o.name, results=[_retype(r.type, epi.width) for r in o.results],
            operands=[vals.get(v, epi.invariant.get(v, v)) for v in o.operands],
            attributes=attrs)
        for old, r in zip(o.results, new.results):
            vals[old] = r
    transfer_write(vals[epi.stored], epi.d.value, at, fits)


def _build_c_pops(g, c, n_idx, m_idx, last=False, first=False):
    for pop in _c_pop_chunks(g, c, n_idx, m_idx, last, first):
        pop()


def _c_pop_chunks(g, c, n_idx, m_idx, last=False, first=False):
    """Each pop of one subtile added into C -- written there where `first` holds -- as emitters
    returning the rows they pop; where `last` holds the epilogue then runs on the sum. Both flags
    are a bool or an i1."""
    from mlir.dialects import arith, scf

    def pop(i):
        vpop = bitcast(
            vcix("vcix.v.i", [g.rvl_acc], [c.word],
                  {"opcode": g.a64(2), "imm": g.a64(0),
                   "rs2": g.a64(0)}).results[0], c.vec)
        at = _build_spad_at(g, n_idx, m_idx, i, g.M_val, g.push_val)
        grid, _, M, rows, _ = g.reach
        fits = _in_bank(c, grid[0], M, grid[2], rows, i)

        def settle(total, epilogue=True):
            transfer_write(total, c.value, at, fits)
            if g.epi is None or not epilogue:
                return
            if last is True:
                _build_epilogue(g.epi, total, at, fits)
            elif last is not False:
                branch = scf.IfOp(last)
                with ir.InsertionPoint(branch.then_block):
                    _build_epilogue(g.epi, total, at, fits)
                    scf.YieldOp([])

        def accumulate():
            settle(arith.AddFOp(transfer_read(c.vec, c.value, at, c.pad, fits), vpop).result)
        if first is True:
            settle(vpop)
        elif first is False:
            accumulate()
        else:
            branch = scf.IfOp(first, has_else=True)
            with ir.InsertionPoint(branch.then_block):
                settle(vpop, g.arith or grid[1] == 1)
                scf.YieldOp([])
            with ir.InsertionPoint(branch.else_block):
                accumulate()
                scf.YieldOp([])
        return g.nr_acc
    return [functools.partial(pop, i) for i in range(0, g.M_LOOP, g.nr_acc)]


def _in_bank(operand, outer_trips, extent, inner_trips, span, at):
    """Whether every access `_build_spad_at` makes at `at` stays inside the operand's bank: its last
    subtile's offset plus one vector reaches no further than the bank holds."""
    last = (outer_trips - 1) * extent + (inner_trips - 1) * span + at
    return last + ir.VectorType(operand.vec).shape[0] <= ir.MemRefType(operand.value.type).shape[-1]


def _build_spad_at(g, outer, inner, at, extent, span):
    """[0, outer * extent + inner * span + at]: the bank map applied, or the same sum in index
    arithmetic for a stream whose subtile coordinates no affine map can state."""
    if not g.arith:
        return [const_index(0),
                _build_apply(g.spad_map, [outer, inner, const_index(at), extent, span])]
    from mlir.dialects import arith
    flat = arith.AddIOp(arith.MulIOp(outer, extent).result,
                        arith.MulIOp(inner, span).result).result
    return [const_index(0), arith.AddIOp(flat, const_index(at)).result]


def _get_spad_map():
    d0, d1, d2 = (ir.AffineDimExpr.get(i) for i in range(3))
    s0, s1 = (ir.AffineSymbolExpr.get(i) for i in range(2))
    return ir.AffineMap.get(3, 2, [d0 * s0 + d1 * s1 + d2])


def _build_apply(map_, operands):
    from mlir.dialects import affine
    return affine.AffineApplyOp(map_, list(operands)).result
