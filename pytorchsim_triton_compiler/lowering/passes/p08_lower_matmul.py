"""torchsim-lower-matmul: every torchsim.dot onto the systolic array -- per subtile the array takes,
B's rows pushed, A's rows pushed, one compute, the result popped and added into C; with the queue
size known, the subtiles stream, each pushed while the one `depth` before it is popped."""

import functools
import sys
from collections import namedtuple

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from .lib_bank_layout import bank_view, index_bank_casts  # noqa: E402
from .lib_ir import (DOT_OP, array_kind, build_trip_loop, const_index, elem_bits,  # noqa: E402
                     op_of, transfer_read, transfer_write, zero_of)
from .lib_indexing_maps import MATMUL_K, MATMUL_M, MATMUL_N, matmul_axis  # noqa: E402
from .lib_vcix import bitcast, i64 as a64, vcix, word_vector  # noqa: E402

MATMUL_OPS = (DOT_OP,)

FORBIDS = MATMUL_OPS

MARKERS = MATMUL_OPS

_Gemm = namedtuple(
    "_Gemm",
    "spad_map a64 rvl rvl_acc elen nr_element nr_acc M_val push_val M_LOOP")

_GemmOperand = namedtuple("_GemmOperand", "value vec word pad")

class ArrayRefusal(Exception):
    """The array cannot take this matmul."""


def run(module, *, vectorlane, vlen, queue_entries, **_):
    """Every matmul in `module` lowered in place; `queue_entries` is the array's input and output
    queue size."""
    from .lib_ir import walk
    index_bank_casts(module)
    for o in [o for o in walk(module.operation) if o.name in MATMUL_OPS]:
        _rewrite_matmul(o, vectorlane, vlen, queue_entries)
    return module


def _rewrite_matmul(op, SS, vlen, queue_entries):
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

    g = _Gemm(bank_map, a64, rvl, rvl_acc, elen, nr_element, nr_acc,
              M_val, push_val, M_LOOP)
    b = _GemmOperand(bankB, vectorType, wordType, zero_pad)
    a = _GemmOperand(bankA, vectorMType, wordMType, zero_pad)
    c = _GemmOperand(bankC, vectorCType, wordCType, acc_pad)
    grid = (N // sysN, K // sysK, M // push_length)
    depth = _get_stream_depth(g, queue_entries, grid[0] * grid[1] * grid[2])

    if depth > 1:
        padding = (None, None, None)
        if K <= sysK:
            padding = _build_short_k_padding(ip, K, SS, elen, wordType, wordMType)
        _build_stream(ip, g, a, b, c, grid, depth, K, K_val, SS, padding)
        op.erase()
        return

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
        _build_c_pops(g, c, n_idx, m_idx)
    op.erase()


def _get_stream_depth(g, queue_entries, subtiles):
    """How many subtiles are pushed ahead of their pops: as many as the input and output queues
    hold together, so no push waits on a pop behind it; 1, the plain order, when a subtile pops
    other than the rows it pushed."""
    pushed, popped = _rows(g.M_LOOP, g.nr_element), _rows(g.M_LOOP, g.nr_acc)
    if pushed != popped:
        return 1
    return max(1, min(subtiles, 2 * queue_entries // pushed))


def _rows(extent, per_insn):
    return -(-extent // per_insn) * per_insn


def _build_stream(ip, g, a, b, c, grid, depth, K, K_val, SS, padding):
    """The `grid` (n, kk, m) subtiles as one stream: subtile t's rows go in between the pops of
    t - depth, popped rows kept ahead of pushed ones, and new weights follow the last push of the
    old. The first `depth` subtiles and the last pops are unrolled; the rest is one loop."""
    total = grid[0] * grid[1] * grid[2]
    with ip:
        for t in range(depth):
            _build_subtile(g, a, b, c, grid, t, None, K, K_val, SS, padding)
    if total > depth:
        iv, body = build_trip_loop(ip, total - depth)
        with body:
            t = _build_apply(ir.AffineMap.get(1, 0, [ir.AffineDimExpr.get(0) + depth]), [iv])
            _build_subtile(g, a, b, c, grid, t, iv, K, K_val, SS, padding)
    with ip:
        for t in range(total - depth, total):
            for pop in _c_pop_chunks(g, c, *_coords(grid, t)[::2]):
                pop()


def _build_subtile(g, a, b, c, grid, t, drained, K, K_val, SS, padding):
    """Subtile t: its weights if they change here, its pushes after each pop of subtile
    `drained` (an index, or None) that keeps the pops ahead, then the rest of those pops."""
    from mlir.dialects import arith, scf
    zero_vector, a_zero, live_lane = padding
    n, kk, m = _coords(grid, t)
    if isinstance(t, int):
        if t % grid[2] == 0:
            _build_b_pushes(g, b, n, kk, K, K_val, SS, zero_vector)
    elif grid[2] == 1:
        _build_b_pushes(g, b, n, kk, K, K_val, SS, zero_vector)
    else:
        first = scf.IfOp(arith.CmpIOp(arith.CmpIPredicate.eq, m, const_index(0)).result)
        with ir.InsertionPoint(first.then_block):
            _build_b_pushes(g, b, n, kk, K, K_val, SS, zero_vector)
            scf.YieldOp([])
    drain = [] if drained is None else _c_pop_chunks(g, c, *_coords(grid, drained)[::2])
    pushed = popped = 0
    for rows, push in _a_push_chunks(g, a, kk, m, live_lane, a_zero):
        while drain and popped < pushed + rows:
            popped += drain.pop(0)()
        push()
        pushed += rows
    _build_compute(g)
    for pop in drain:
        pop()


def _coords(grid, t):
    """(n, kk, m) of subtile t of an (N, K, M) grid, m innermost: constants for a constant t."""
    _, nk, nm = grid
    if isinstance(t, int):
        return tuple(const_index(v) for v in (t // (nk * nm), t // nm % nk, t % nm))
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
            f"different element types; the array is pushed with a single sew "
            f"and cannot be told to read the two planes differently")
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


def _build_b_pushes(g, b, n_idx, k_idx, K, K_val, SS, zero_vector):
    for i in range(0, SS, g.nr_element):
        if i < K:
            at = _build_spad_at(g.spad_map, n_idx, k_idx, i, K_val, const_index(SS))
            wv = bitcast(transfer_read(b.vec, b.value, at, b.pad), b.word)
        else:
            wv = zero_vector
        vcix("vcix.iv", [wv, g.rvl], [],
              {"opcode": g.a64(1), "imm": g.a64(0), "rd": g.a64(0)})


def _build_a_pushes(g, a, k_idx, m_idx, live_lane, a_zero):
    for _, push in _a_push_chunks(g, a, k_idx, m_idx, live_lane, a_zero):
        push()


def _a_push_chunks(g, a, k_idx, m_idx, live_lane, a_zero):
    """Each input push of one subtile, as (rows, emitter), in order."""
    from mlir.dialects import arith

    def push(i):
        at = _build_spad_at(g.spad_map, k_idx, m_idx, i, g.M_val, g.push_val)
        iv = bitcast(transfer_read(a.vec, a.value, at, a.pad), a.word)
        if live_lane is not None:
            iv = arith.SelectOp(live_lane, iv, a_zero).result
        vcix("vcix.iv", [iv, g.rvl], [],
              {"opcode": g.a64(0), "imm": g.a64(0), "rd": g.a64(0)})
        return g.nr_element
    return [(g.nr_element, functools.partial(push, i)) for i in range(0, g.M_LOOP, g.nr_element)]


def _build_compute(g):
    vcix("vcix.i", [g.rvl], [],
          {"opcode": g.a64(1), "imm": g.a64(4), "rd": g.a64(0),
           "rs2": g.a64(0), "sew": g.a64(g.elen), "lmul": g.a64(0)})


def _build_c_pops(g, c, n_idx, m_idx):
    for pop in _c_pop_chunks(g, c, n_idx, m_idx):
        pop()


def _c_pop_chunks(g, c, n_idx, m_idx):
    """Each pop of one subtile added into C, as emitters returning the rows they pop, in order."""
    from mlir.dialects import arith

    def pop(i):
        vpop = bitcast(
            vcix("vcix.v.i", [g.rvl_acc], [c.word],
                  {"opcode": g.a64(2), "imm": g.a64(0),
                   "rs2": g.a64(0)}).results[0], c.vec)
        at = _build_spad_at(g.spad_map, n_idx, m_idx, i, g.M_val, g.push_val)
        prev = transfer_read(c.vec, c.value, at, c.pad)
        transfer_write(arith.AddFOp(prev, vpop).result, c.value, at)
        return g.nr_acc
    return [functools.partial(pop, i) for i in range(0, g.M_LOOP, g.nr_acc)]


def _build_spad_at(bank_map, outer, inner, at, extent, span):
    return [const_index(0),
            _build_apply(bank_map, [outer, inner, const_index(at), extent, span])]


def _get_spad_map():
    d0, d1, d2 = (ir.AffineDimExpr.get(i) for i in range(3))
    s0, s1 = (ir.AffineSymbolExpr.get(i) for i in range(2))
    return ir.AffineMap.get(3, 2, [d0 * s0 + d1 * s1 + d2])


def _build_apply(map_, operands):
    from mlir.dialects import affine
    return affine.AffineApplyOp(map_, list(operands)).result
