"""torchsim-lower-matmul: every torchsim.dot onto the systolic array -- B's rows pushed, A's rows
pushed, one compute, the result popped and added into C, per subtile the array takes."""

import sys
from collections import namedtuple

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from .lib_bank_layout import bank_view, index_bank_casts  # noqa: E402
from .lib_ir import (DOT_OP, array_kind, build_trip_loop, const_index, elem_bits,  # noqa: E402
                     int_array, op_of, transfer_read, transfer_write, zero_of)
from .lib_indexing_maps import MATMUL_K, MATMUL_M, MATMUL_N, matmul_axis  # noqa: E402
from .lib_vcix import bitcast, i64 as a64, vcix, word_vector  # noqa: E402

MATMUL_OPS = (DOT_OP,)

FORBIDS = MATMUL_OPS

MARKERS = MATMUL_OPS

_Gemm = namedtuple(
    "_Gemm",
    "spad_map a64 rvl rvl_acc elen nr_element nr_acc M_val push_val M_LOOP")

_GemmOperand = namedtuple("_GemmOperand", "value vec word pad")

_Split = namedtuple("_Split", "base axis numel")


class ArrayRefusal(Exception):
    """The array cannot take this matmul."""


def run(module, *, vectorlane, vlen, **_):
    """Every matmul in `module` lowered in place."""
    from .lib_ir import walk
    index_bank_casts(module)
    for o in [o for o in walk(module.operation) if o.name in MATMUL_OPS]:
        _rewrite_matmul(o, vectorlane, vlen, _split_operand_loads(o, vectorlane))
    return module


def _split_operand_loads(op, SS):
    M, N, K = _get_mnk(op)
    subs = {MATMUL_M: min(M, SS), MATMUL_N: min(N, SS), MATMUL_K: min(K, SS)}
    stored = {0: (MATMUL_K, MATMUL_M), 1: (MATMUL_N, MATMUL_K), 2: (MATMUL_N, MATMUL_M)}
    plans = {}
    for k, dims in stored.items():
        buf = op.operands[k]
        tx = _get_simple_load(op, buf)
        shape = list(ir.MemRefType(buf.type).shape)
        if tx is None or len(shape) != 2 or matmul_axis(op, k, dims[0]) != 0:
            continue
        waits = [op_of(u.owner) for u in tx.opview.tag_idx.uses
                 if op_of(u.owner).name == "torchsim.wait"]
        ds = int_array(tx.opview.dram_stride)
        axis = 0 if ds[0] > ds[1] else 1
        sub = subs[dims[axis]]
        if len(waits) == 1 and _is_waiting_on(waits[0], op) and shape[axis] // sub > 1:
            plans[k] = (tx, waits[0], axis, sub, shape, ds)
    splits, base = {}, _get_next_slot(op)
    for k, (_tx, _w, axis, sub, shape, _ds) in sorted(plans.items()):
        splits[k] = _Split(base, axis, sub * shape[1 - axis])
        base += shape[axis] // sub
    if splits:
        last = max((p[0] for p in plans.values()), key=lambda t: sum(
            u[0].operation.is_before_in_block(t.operation) for u in plans.values()))
        for k, piece in _get_first_use_order(M, N, K, subs, splits):
            _build_piece(last, plans[k], splits[k], piece, SS)
        for tx, wait, *_ in plans.values():
            wait.operation.erase()
            tx.operation.erase()
    return splits


def _get_first_use_order(M, N, K, subs, splits):
    seen = []

    def use(k, outer, inner):
        if k in splits:
            got = (k, outer if splits[k].axis == 0 else inner)
            if got not in seen:
                seen.append(got)

    for n in range(N // subs[MATMUL_N]):
        for kk in range(K // subs[MATMUL_K]):
            use(1, n, kk)
            for m in range(M // subs[MATMUL_M]):
                use(0, kk, m)
                use(2, n, m)
    return seen


def _build_piece(at, plan, split, i, SS):
    from mlir.dialects import arith
    from mlir.ir import ArrayAttr, InsertionPoint, IntegerAttr
    from ...dialects import torchsim
    tx, _wait, axis, sub, shape, ds = plan
    view = tx.opview
    with InsertionPoint(at):
        moved = {"dram_idx": arith.AddIOp(view.dram_idx, const_index(i * sub * ds[axis])).result,
                 "sram_idx": arith.AddIOp(view.sram_idx, const_index(
                     (i * sub // SS) * shape[1] if axis == 0 else i * sub)).result,
                 "tag_idx": const_index(split.base + i)}
        piece = tx.operation.clone()
    for name, v in moved.items():
        piece.operands[torchsim.operand_number(piece, name)] = v
    size = [sub, shape[1]] if axis == 0 else [shape[0], sub]
    piece.attributes["subtile_size"] = ArrayAttr.get(
        [IntegerAttr.get(ir.IntegerType.get_signless(64), v) for v in size])


def _get_simple_load(op, buf):
    loads = [op_of(u.owner) for u in buf.uses
             if op_of(u.owner).name == "torchsim.transfer"
             and "MVIN" in op_of(u.owner).opview.dma_kind.value]
    if len(loads) != 1:
        return None
    tx = loads[0]
    v = tx.opview
    plain = (v.index is None and v.masked_axes is None and v.dyn_stride_axes is None
             and v.subtile_size is None and v.indirect_dim is None and v.cut is None
             and v.async_ is not None)
    same_block = tx.operation.block == op.operation.block and tx.operation.is_before_in_block(op.operation)
    return tx if plain and same_block else None


def _is_waiting_on(wait, op):
    ops = [o.operation for o in wait.operation.block.operations]
    at = next(i for i, o in enumerate(ops) if o == wait.operation)
    rest = [o for o in ops[at + 1:] if o.name != "torchsim.wait"]
    return bool(rest) and rest[0] == op.operation


def _get_next_slot(op):
    from mlir.ir import IntegerAttr
    from .lib_ir import walk
    from ...dialects import torchsim
    f = op.operation
    while f.name != "func.func":
        f = f.parent
    top = -1
    for t in [o for o in walk(f) if o.name == "torchsim.transfer"]:
        v = op_of(t.operands[torchsim.operand_number(t, "tag_idx")].owner)
        if v is not None and v.name == "arith.constant":
            top = max(top, IntegerAttr(v.attributes["value"]).value)
    return top + 1


def _rewrite_matmul(op, SS, vlen, splits):
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

    g = _Gemm(bank_map, a64, rvl, rvl_acc, elen, nr_element, nr_acc,
              M_val, push_val, M_LOOP)
    b = _GemmOperand(bankB, vectorType, wordType, zero_pad)
    a = _GemmOperand(bankA, vectorMType, wordMType, zero_pad)
    c = _GemmOperand(bankC, vectorCType, wordCType, acc_pad)

    with body_ip:
        if 1 in splits:
            _build_subtile_wait(splits[1], n_idx, k_idx)
        _build_b_pushes(g, b, n_idx, k_idx, K, K_val, SS, zero_vector)

    m_idx = c0
    if M > push_length:
        m_idx, body_ip = build_trip_loop(body_ip, M // push_length)

    with body_ip:
        if 0 in splits:
            _build_subtile_wait(splits[0], k_idx, m_idx)
        if 2 in splits:
            _build_subtile_wait(splits[2], n_idx, m_idx)
        _build_a_pushes(g, a, k_idx, m_idx, live_lane, a_zero)
        _build_compute(g)
        _build_c_pops(g, c, n_idx, m_idx)
    op.erase()


def _build_subtile_wait(split, outer, inner):
    from ...dialects import torchsim
    d0, d1 = ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)
    slot = _build_apply(ir.AffineMap.get(2, 0, [d0 + d1]),
                        [outer if split.axis == 0 else inner, const_index(split.base)])
    torchsim.WaitOp(slot, const_index(split.numel))


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
    from mlir.dialects import arith
    for i in range(0, g.M_LOOP, g.nr_element):
        at = _build_spad_at(g.spad_map, k_idx, m_idx, i, g.M_val, g.push_val)
        iv = bitcast(transfer_read(a.vec, a.value, at, a.pad), a.word)
        if live_lane is not None:
            iv = arith.SelectOp(live_lane, iv, a_zero).result
        vcix("vcix.iv", [iv, g.rvl], [],
              {"opcode": g.a64(0), "imm": g.a64(0), "rd": g.a64(0)})


def _build_compute(g):
    vcix("vcix.i", [g.rvl], [],
          {"opcode": g.a64(1), "imm": g.a64(4), "rd": g.a64(0),
           "rs2": g.a64(0), "sew": g.a64(g.elen), "lmul": g.a64(0)})


def _build_c_pops(g, c, n_idx, m_idx):
    from mlir.dialects import arith
    for i in range(0, g.M_LOOP, g.nr_acc):
        vpop = bitcast(
            vcix("vcix.v.i", [g.rvl_acc], [c.word],
                  {"opcode": g.a64(2), "imm": g.a64(0),
                   "rs2": g.a64(0)}).results[0], c.vec)
        at = _build_spad_at(g.spad_map, n_idx, m_idx, i, g.M_val, g.push_val)
        prev = transfer_read(c.vec, c.value, at, c.pad)
        transfer_write(arith.AddFOp(prev, vpop).result, c.value, at)


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
