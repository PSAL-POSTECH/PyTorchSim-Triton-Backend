"""torchsim-lower-matmul-mxu: every torchsim.dot onto the matrix unit of a target whose
`matmul_unit` is "mxu" (vcix-accelerator's libtpu_mxu.so): an array twice as wide and as deep as
the machine has lanes, so an operand is two registers -- a vmatprep with the upper lane tile, then
the vmatpush (weights) or the vmatmul (inputs) with the lower. A word of 32 bits is one f32 or two
rows of 16 bits; a result leaves as f32, the lower lane tile's columns and the upper's in turn,
and each pop is added into C. Runs in place of torchsim-lower-matmul, which it leaves as it is."""
import sys

from ... import toolchain

toolchain.activate_bindings(sys.path)

import mlir.ir as ir  # noqa: E402

from . import p08_lower_matmul as p08  # noqa: E402
from .lib_bank_layout import bank_view, index_bank_casts  # noqa: E402
from .lib_ir import (build_trip_loop, const_index, elem_bits, transfer_read,  # noqa: E402
                     transfer_write, zero_of)
from .lib_vcix import bitcast, i64 as a64, vcix  # noqa: E402

MATMUL_OPS = p08.MATMUL_OPS

FORBIDS = MATMUL_OPS

MARKERS = MATMUL_OPS

# The vcix.iv / vcix.v.i opcodes of the unit's four instructions, and the bits of their imm.
VMATMUL, VMATPUSH, VMATPREP, VPOP = 0, 1, 2, 2
FORMAT_F32, FORMAT_F16, FORMAT_BF16 = 0, 1, 3
SECOND = 4      # a vmatprep of an input, not of weights; a vpop of the upper columns


def run(module, *, vectorlane, vlen, **_):
    """Every matmul in `module` lowered in place."""
    from .lib_ir import walk
    index_bank_casts(module)
    for o in [o for o in walk(module.operation) if o.name in MATMUL_OPS]:
        _rewrite_matmul(o, vectorlane, vlen)
    return module


def _format(elt):
    if isinstance(elt, ir.BF16Type):
        return FORMAT_BF16
    return {16: FORMAT_F16, 32: FORMAT_F32}[elem_bits(elt)]


def _refuse(op, M, N, K, lanes, rows, elt, acc_elt):
    if elem_bits(elt) not in (16, 32):
        raise p08.ArrayRefusal(
            f"the matrix unit takes words of one f32 or two rows of 16 bits, not {elt}: {op}")
    if elem_bits(acc_elt) != 32:
        raise p08.ArrayRefusal(f"the matrix unit pops f32, and C is {acc_elt}: {op}")
    for d, name in ((N, "N"), (K, "K")):
        if d % lanes and (name == "K" or d > lanes):
            raise p08.ArrayRefusal(
                f"matmul {name}={d} is not whole lane tiles of {lanes}; the matrix unit's "
                f"lowering masks no lanes yet: {op}")
    if M % rows:
        raise p08.ArrayRefusal(
            f"matmul M={M} is not whole registers of {rows} rows; the matrix unit's lowering "
            f"has no shorter last push yet: {op}")


def _rewrite_matmul(op, lanes, vlen):
    from mlir.dialects import arith

    A, B, C = op.operands[0], op.operands[1], op.operands[2]
    mtA, mtB, mtC = (ir.MemRefType(x.type) for x in (A, B, C))
    elt, acc_elt = mtA.element_type, mtC.element_type
    M, N, K = p08._get_mnk(op)
    p08._refuse_unlowerable(op, elt, acc_elt, mtB.element_type)
    words = vlen // 32
    pack = 32 // elem_bits(elt) if elem_bits(elt) in (16, 32) else 0
    rows = words * pack
    _refuse(op, M, N, K, lanes, rows, elt, acc_elt)

    i32 = ir.IntegerType.get_signless(32)
    word_ty = ir.VectorType.get([words], i32)
    u = _Unit(lanes=lanes, words=words, rows=rows, fmt=_format(elt), M=M, K=K,
              vec=ir.VectorType.get([rows], elt), word=word_ty,
              acc=ir.VectorType.get([words], acc_elt))
    ip = ir.InsertionPoint(op)
    with ip:
        u.rvl = arith.ConstantOp(ir.IntegerType.get_signless(64), a64(words)).result
        u.zero = arith.ConstantOp(word_ty, ir.DenseElementsAttr.get_splat(
            word_ty, ir.IntegerAttr.get(i32, 0))).result
        u.a, u.b, u.c = (bank_view(x, lanes) for x in (A, B, C))
        u.pad, u.acc_pad = zero_of(elt), zero_of(acc_elt)

    # Lane tiles go to the array in pairs; an odd last one goes with zeros for its upper half.
    n_tiles, k_tiles = max(1, N // lanes), K // lanes
    for n_first, n_pairs, n_upper in _pairs(n_tiles):
        n_ip, n_tile = _tile_loop(ip, n_first, n_pairs)
        for k_first, k_pairs, k_upper in _pairs(k_tiles):
            k_ip, k_tile = _tile_loop(n_ip, k_first, k_pairs)
            with k_ip:
                _build_weights(u, n_tile, k_tile, n_upper, k_upper)
            m_idx, m_ip = (const_index_at(k_ip, 0), k_ip)
            if M > rows:
                m_idx, m_ip = build_trip_loop(k_ip, M // rows)
            with m_ip:
                _build_input(u, k_tile, m_idx, k_upper)
                _build_pops(u, n_tile, m_idx, n_upper)
    op.erase()


class _Unit:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def const_index_at(ip, v):
    with ip:
        return const_index(v)


def _pairs(tiles):
    """(first tile, trips, whether each trip has an upper tile): the pairs, then an odd last tile."""
    out = []
    if tiles // 2:
        out.append((0, tiles // 2, True))
    if tiles % 2:
        out.append((tiles - 1, 1, False))
    return out


def _tile_loop(ip, first, trips):
    """A loop over `trips` pairs of lane tiles from tile `first`: (the point inside, the lower
    tile's number as an index)."""
    from mlir.dialects import arith
    if trips == 1:
        return ip, const_index_at(ip, first)
    iv, inner = build_trip_loop(ip, trips)
    with inner:
        return inner, arith.AddIOp(arith.MulIOp(iv, const_index(2)).result,
                                   const_index(first)).result


def _at(tile, upper, extent, inner, span, at):
    """[0, (tile + upper) * extent + inner * span + at], the in-bank offset of one register."""
    from mlir.dialects import arith
    if upper:
        tile = arith.AddIOp(tile, const_index(1)).result
    flat = arith.MulIOp(tile, const_index(extent)).result
    if inner is not None:
        flat = arith.AddIOp(flat, arith.MulIOp(inner, const_index(span)).result).result
    return [const_index(0), arith.AddIOp(flat, const_index(at)).result]


def _words(u, bank, at):
    """One register of the operand at `at`, as the words the unit takes."""
    from mlir.dialects import vector
    value = transfer_read(u.vec, bank, at, u.pad, True)
    if u.rows == u.words:
        return bitcast(value, u.word)
    return vector.BitCastOp(u.word, value).result


def _build_weights(u, n_tile, k_tile, n_upper, k_upper):
    """A whole array of weights, 2 x lanes rows of K: B's rows of this pair of K tiles, and zeros
    for an upper K tile there is not. The vmatprep carries the upper N tile's columns."""
    from mlir.dialects import arith
    k_at = arith.MulIOp(k_tile, const_index(u.lanes)).result
    for row in range(0, 2 * u.lanes, u.rows):
        live = row < u.lanes or k_upper
        upper = _words(u, u.b, _at(n_tile, True, u.K, k_at, 1, row)) if live and n_upper else u.zero
        vcix("vcix.iv", [upper, u.rvl], [],
             {"opcode": a64(VMATPREP), "imm": a64(u.fmt), "rd": a64(0)})
        lower = _words(u, u.b, _at(n_tile, False, u.K, k_at, 1, row)) if live else u.zero
        vcix("vcix.iv", [lower, u.rvl], [],
             {"opcode": a64(VMATPUSH), "imm": a64(u.fmt), "rd": a64(0)})


def _build_input(u, k_tile, m_idx, k_upper):
    """One register of rows of A: the vmatprep carries the upper K tile's lanes."""
    upper = _words(u, u.a, _at(k_tile, True, u.M, m_idx, u.rows, 0)) if k_upper else u.zero
    vcix("vcix.iv", [upper, u.rvl], [],
         {"opcode": a64(VMATPREP), "imm": a64(u.fmt | SECOND), "rd": a64(0)})
    lower = _words(u, u.a, _at(k_tile, False, u.M, m_idx, u.rows, 0))
    vcix("vcix.iv", [lower, u.rvl], [],
         {"opcode": a64(VMATMUL), "imm": a64(u.fmt), "rd": a64(0)})


def _build_pops(u, n_tile, m_idx, n_upper):
    """The rows one vmatmul computed, a register of f32 at a time: the lower N tile's columns, then
    the upper's, each added into C. The upper half of an odd last tile is popped and dropped."""
    from mlir.dialects import arith
    for row in range(0, u.rows, u.words):
        for upper in (False, True):
            got = bitcast(vcix("vcix.v.i", [u.rvl], [u.word],
                               {"opcode": a64(VPOP), "imm": a64(SECOND if upper else 0),
                                "rs2": a64(0)}).results[0], u.acc)
            if upper and not n_upper:
                continue
            at = _at(n_tile, upper, u.M, m_idx, u.rows, row)
            total = arith.AddFOp(transfer_read(u.acc, u.c, at, u.acc_pad, True), got).result
            transfer_write(total, u.c, at, True)
