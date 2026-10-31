"""gather_in_bank: a `pytorchsim.gather` along the axis inside each bank -- each bank's row gathered
in its registers by its index row (vrgather.vv); nothing crosses banks (lower_gather's axis other than 0)."""

from ...passes.utils.bank_layout import bank_depth, bank_view
from ...passes.utils.ir_utils import const_index, elem_bits, transfer_read, transfer_write, zero_of
from ...passes.utils.register_groups import GROUPS, integer_view, register_group

import mlir.ir as ir  # noqa: E402


def _refuse(op, why):
    raise NotImplementedError(
        f"pytorchsim.gather cannot be lowered inside the banks ({why}); "
        f"materialize_layouts must not mint one this cannot take: {op}")


def lower_gather_in_bank(op, lanes, vlen, lmul):
    """A pytorchsim.gather along the one axis inside each lane's bank: each lane's row of the source
    gathered in its registers by the index row (vrgather.vv, at the wider of the element and the index,
    every index on the row it came from), into its row -- no lane crosses; past a group, `_gather_by_groups`."""
    from mlir.dialects import arith, llvm
    from mlir.ir import Operation
    src, index, dst = op.operands[0], op.operands[1], op.operands[2]
    if any(ir.MemRefType(v.type).rank != 2 for v in (src, index, dst)) or ir.IntegerAttr(
            op.attributes["axis"]).value != 1:
        return _refuse(op, "a gather inside the lanes other than along a [lane, depth] "
                                "tile's depth")
    depth = bank_depth(dst, lanes)
    span = ir.MemRefType(src.type).shape[1]
    if depth is None or bank_depth(src, lanes) != depth or bank_depth(index, lanes) != depth or any(
            ir.MemRefType(v.type).shape[1] != span for v in (index, dst)):
        return _refuse(op, "its source, index and destination are not one bank shape")
    if span < 1 or depth % span:
        return _refuse(op, f"a lane's bank of {depth} elements is no whole number of the gathered "
                                f"axis' {span}, so an index of it names no row of its own")
    elt = ir.MemRefType(src.type).element_type
    ielt = ir.MemRefType(index.type).element_type
    ity, there, back = integer_view(elt)
    i64 = ir.IntegerType.get_signless(64)
    if depth > span and depth > 1 << (ity.width - 1):
        return _refuse(op, f"{depth // span} rows in one lane's bank reach element {depth - 1}, past "
                                f"what the i{ity.width} this gather reads its index at can name")
    wide = max(ity.width, elem_bits(ielt))
    if depth * wide > GROUPS[-1] * vlen:
        cap = lmul * vlen // wide
        if depth % cap:
            return _refuse(op, f"a lane's row of {depth} is no whole number of "
                                    f"{cap}-element register groups")
        with ir.InsertionPoint(op):
            _gather_by_groups(src, index, dst, depth, cap, span, lanes, lmul)
        op.erase()
        return True
    whole, m = register_group(depth, wide, vlen)
    with ir.InsertionPoint(op):
        src_bank, idx_bank, dst_bank = (bank_view(v, lanes) for v in (src, index, dst))
        row = ir.VectorType.get([depth], elt)
        v = transfer_read(row, src_bank, [const_index(0), const_index(0)], zero_of(elt), True)
        k = transfer_read(ir.VectorType.get([depth], ielt), idx_bank, [const_index(0), const_index(0)],
                          zero_of(ielt), True)
        iv = ir.VectorType.get([depth], ity)
        if there:
            v = there(iv, v).result
        if ielt != ity:
            k = (arith.ExtSIOp if elem_bits(ielt) < ity.width else arith.TruncIOp)(iv, k).result
        if depth > span:
            k = arith.AddIOp(k, _build_row_bases(iv, 0, span)).result
        sc = ir.VectorType.get([m * 64 // ity.width], ity, scalable=[True])
        poison = llvm.PoisonOp(sc).result
        put = lambda x: Operation.create("llvm.intr.vector.insert", results=[sc], operands=[poison, x],
                                         attributes={"pos": ir.IntegerAttr.get(i64, 0)}).results[0]
        got = llvm.call_intrinsic(sc, "llvm.riscv.vrgather.vv",
                                  [poison, put(v), put(k), arith.ConstantOp(i64, depth).result], [], [])
        got = Operation.create("llvm.intr.vector.extract", results=[iv], operands=[got],
                               attributes={"pos": ir.IntegerAttr.get(i64, 0)}).results[0]
        if back:
            got = back(row, got).result
        transfer_write(got, dst_bank, [const_index(0), const_index(0)], True)
    op.erase()
    return True


def _gather_by_groups(src, index, dst, depth, cap, span, lanes, lmul):
    """A lane's row of `depth` gathered `cap` elements (one LMUL-`lmul` group) at a time: each piece of the
    destination takes, from every piece of the source, the elements its index falls in (vrgather.vv), each
    index first moved onto the row of `span` it was read from."""
    from mlir.dialects import arith, llvm
    from mlir.ir import Operation
    elt = ir.MemRefType(src.type).element_type
    ielt = ir.MemRefType(index.type).element_type
    ity, there, back = integer_view(elt)
    i64 = ir.IntegerType.get_signless(64)
    src_bank, idx_bank, dst_bank = (bank_view(v, lanes) for v in (src, index, dst))
    iv, row = ir.VectorType.get([cap], ity), ir.VectorType.get([cap], elt)
    sc = ir.VectorType.get([lmul * 64 // ity.width], ity, scalable=[True])
    splat = lambda n: arith.ConstantOp(iv, ir.DenseElementsAttr.get_splat(iv, ir.IntegerAttr.get(ity, n))).result
    put = lambda x: Operation.create("llvm.intr.vector.insert", results=[sc], operands=[llvm.PoisonOp(sc).result, x],
                                     attributes={"pos": ir.IntegerAttr.get(i64, 0)}).results[0]
    at = lambda j: [const_index(0), const_index(j * cap)]
    for g in range(depth // cap):
        k = transfer_read(ir.VectorType.get([cap], ielt), idx_bank, at(g), zero_of(ielt), True)
        if ielt != ity:
            k = (arith.ExtSIOp if elem_bits(ielt) < ity.width else arith.TruncIOp)(iv, k).result
        if depth > span:
            k = arith.AddIOp(k, _build_row_bases(iv, g * cap, span)).result
        acc = splat(0)
        for j in range(depth // cap):
            v = transfer_read(row, src_bank, at(j), zero_of(elt), True)
            if there:
                v = there(iv, v).result
            lo, hi = splat(j * cap), splat((j + 1) * cap)
            mine = arith.AndIOp(arith.CmpIOp(arith.CmpIPredicate.sge, k, lo).result,
                                arith.CmpIOp(arith.CmpIPredicate.slt, k, hi).result).result
            got = llvm.call_intrinsic(sc, "llvm.riscv.vrgather.vv",
                                      [llvm.PoisonOp(sc).result, put(v), put(arith.SubIOp(k, lo).result),
                                       arith.ConstantOp(i64, cap).result], [], [])
            got = Operation.create("llvm.intr.vector.extract", results=[iv], operands=[got],
                                   attributes={"pos": ir.IntegerAttr.get(i64, 0)}).results[0]
            acc = arith.SelectOp(mine, got, acc).result
        transfer_write(back(row, acc).result if back else acc, dst_bank, at(g), True)


def _build_row_bases(ty, start, span):
    """Where each element of a `ty`-wide run beginning at bank offset `start` has its own row of `span`
    begin: the constant a row's own index is moved by to name an element of that row and no other."""
    from mlir.dialects import arith
    from mlir.ir import DenseElementsAttr, IntegerAttr, VectorType
    elt, n = VectorType(ty).element_type, VectorType(ty).shape[0]
    return arith.ConstantOp(ty, DenseElementsAttr.get(
        [IntegerAttr.get(elt, (start + t) // span * span) for t in range(n)], type=ty)).result
