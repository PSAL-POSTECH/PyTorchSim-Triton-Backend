"""dep_analysis.py -- the buffers each TOG node reads and writes, and the edges they make."""
import sys

from ....dialects.torchsim import TransferOp, is_load
from .build_tog import TogBuilder, ir, _reset_ids
from . import build_skeleton as _bs


def _carried_into(arg):
    block = arg.owner
    op = getattr(block, "owner", None)
    op = getattr(op, "operation", op)
    if op is None or not isinstance(op, ir.Operation):
        return None
    if op.name == "scf.while" and len(op.regions) > 1 and \
            any(b == block for b in op.regions[1].blocks):
        term = op.regions[0].blocks[0].operations[
            len(op.regions[0].blocks[0].operations) - 1]
        term = term.operation
        if term.name != "scf.condition":
            return None
        i = arg.arg_number + 1
        src = term.operands[i] if i < len(term.operands) else None
    else:
        i = arg.arg_number - (len(block.arguments) - len(op.operands))
        src = op.operands[i] if 0 <= i < len(op.operands) else None
    return src if src is not None and src.type == arg.type else None


def _global_of(memref_val):
    if isinstance(memref_val, ir.BlockArgument):
        src = _carried_into(memref_val)
        return _global_of(src) if src is not None else None
    owner = memref_val.owner
    op = owner if isinstance(owner, ir.Operation) else getattr(owner, "operation", None)
    if op is None:
        return None
    if op.name == "memref.get_global":
        return str(op.attributes["name"]).strip('@" ')
    if op.operands:
        try:
            return _global_of(op.operands[0])
        except Exception:
            return None
    return None


_LOAD_OPS = {"vector.transfer_read", "affine.vector_load", "vector.load",
             "memref.load", "affine.load", "torchsim.strided_load"}
_STORE_OPS = {"vector.transfer_write", "affine.vector_store", "vector.store",
              "memref.store", "affine.store", "torchsim.strided_store"}
_IGNORE_OPS = {"memref.dealloc", "memref.extract_aligned_pointer_as_index",
               "scf.for", "scf.while", "affine.for", "scf.if",
               "scf.yield", "affine.yield", "scf.condition", "func.return"}


def _is_memref(v):
    try:
        return isinstance(v.type, ir.MemRefType)
    except Exception:
        return False


def _walk_compute_ops(cn):
    for top in cn.operations:
        stack = [top]
        while stack:
            op = stack.pop()
            yield op
            for region in op.operation.regions:
                for block in region.blocks:
                    stack.extend(block.operations)


def _rw_buffers_of_compute(cn):
    reads, writes = {}, {}
    def rd(v):
        b = _global_of(v)
        if b:
            reads.setdefault(b)
    def wr(v):
        b = _global_of(v)
        if b:
            writes.setdefault(b)
    for op in _walk_compute_ops(cn):
        if any(_is_memref(r) for r in op.results):
            continue
        mrefs = [v for v in op.operands if _is_memref(v)]
        if not mrefs:
            continue
        name = op.name
        if name in _LOAD_OPS:
            for v in mrefs:
                rd(v)
        elif name in _STORE_OPS:
            for v in mrefs:
                wr(v)
        elif name == "memref.copy":
            rd(mrefs[0])
            wr(mrefs[-1])
        elif name == TransferOp.OPERATION_NAME:
            tx = op.operation.opview
            dram, sram = tx.dram, tx.sram
            if is_load(op):
                rd(dram)
                wr(sram)
            else:
                rd(sram)
                wr(dram)
        elif name.startswith("linalg."):
            for v in op.inputs:
                if _is_memref(v):
                    rd(v)
            for v in op.outputs:
                if _is_memref(v):
                    rd(v)
                    wr(v)
        elif name in _IGNORE_OPS:
            continue
        else:
            raise RuntimeError(
                f"dep_analysis: unclassified memref op '{name}' in a compute node -- "
                f"it touches an SRAM buffer; classify it in _LOAD_OPS/_STORE_OPS")
    return list(reads), list(writes)


def _dma_buffer(builder, dma_node):
    try:
        f = builder._dma_start_fields(dma_node.op)
    except Exception:
        return None
    val = f["dst"] if not dma_node.is_write else f["src"]
    return _global_of(val)


def compute_buffers(cn, grouper):
    """(read_buffers, write_buffers) for one compute node: the scratchpad globals its ops touch,
    and the virtual buffers its grouping adds to order it against the others."""
    reads, writes = _rw_buffers_of_compute(cn)
    return grouper.buffers(cn, reads, writes)


def analyze(module):
    """(nodes, edges) of the TOG: each node's buffers, and an edge from a reader
    to the last writer of what it reads."""
    _reset_ids()
    builder = TogBuilder()
    _bs._build(module, builder)

    nodes = []
    dma_nodes = [dn for dn in dict.fromkeys(_bs._collect_dma_nodes(builder).values())
                 if hasattr(dn, "is_write")]
    for dn in dma_nodes:
        buf = _dma_buffer(builder, dn)
        nodes.append({
            "kind": "STORE" if dn.is_write else "LOAD",
            "buf": buf, "arg": str(dn.base_addr),
            "reads": [buf] if dn.is_write else [],
            "writes": [buf] if not dn.is_write else [],
            "node": dn,
        })
    for cn in builder.compute_nodes:
        if not cn.operations:
            continue
        creads, cwrites = compute_buffers(cn, builder.grouper)
        nodes.append({
            "kind": builder.grouper.type_names[cn.compute_type],
            "reads": creads,
            "writes": cwrites,
            "node": cn,
            "compute_type": cn.compute_type,
        })

    pos = {}
    idx = [0]
    def _index(op):
        pos[op] = idx[0]; idx[0] += 1
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    _index(o)
    _index(module.operation)
    def _key(n):
        node = n["node"]
        op = getattr(node, "op", None) or (node.operations[0] if getattr(node, "operations", None) else None)
        return pos.get(op, 1 << 30)
    nodes.sort(key=_key)

    edges = []
    last_writer = {}
    for i, n in enumerate(nodes):
        for b in sorted(n["reads"]):
            if b in last_writer:
                edges.append((i, last_writer[b], f"reads {b}"))
        for b in n["writes"]:
            last_writer[b] = i
    return nodes, edges


def _main():
    """Command line: print the nodes and edges of one post-vcix .mlir."""
    path = sys.argv[1]
    ctx = ir.Context(); ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(open(path).read(), ctx)
        nodes, edges = analyze(module)
    print("=== nodes ===")
    for i, n in enumerate(nodes):
        r = ",".join(sorted(n["reads"])) or "-"
        w = ",".join(sorted(n["writes"])) or "-"
        print(f"  #{i:<2} {n['kind']:<16} reads[{r}] writes[{w}]")
    print("=== edges (consumer -> producer) ===")
    for c, p, why in edges:
        print(f"  #{c} ({nodes[c]['kind']}) -> #{p} ({nodes[p]['kind']})   [{why}]")


if __name__ == "__main__":
    _main()
