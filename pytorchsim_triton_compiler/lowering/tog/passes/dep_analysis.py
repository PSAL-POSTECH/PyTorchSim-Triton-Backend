"""dep_analysis.py -- the buffers each TOG node reads and writes, and the edges they make."""
import sys

from ....dialects import torchsim
from ....dialects.torchsim import TransferOp, is_load
from ...passes.lib_bank_layout import in_scratchpad, is_a_scratchpad_buffer
from ...passes.lib_ir import ends_lifetime
from .build_tog import CORE, COMPUTE_TYPE_NAME, TogBuilder, ir, _reset_ids
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


_FOR_LOOPS = ("scf.for", "affine.for")


def _root(v, assumed=()):
    """The buffer (is_a_scratchpad_buffer) `v` names through views and a for loop's carried init,
    in its body and out of it; else the value it stops at. `assumed`: (block argument, root) pairs
    already resolved."""
    while True:
        if isinstance(v, ir.BlockArgument):
            got = [r for a, r in assumed if a == v]
            if got:
                return got[0]
            src = _carried_into(v)
            if src is None:
                return v
            root = _root(src, assumed)
            _refuse_another_buffer_carried_back(v, root, assumed)
            return root
        if is_a_scratchpad_buffer(v):
            return v
        loop = v.owner.operation
        if loop.name in _FOR_LOOPS:
            k = v.result_number
            arg = loop.regions[0].blocks[0].arguments[k + 1]
            root = _root(loop.operands[len(loop.operands) - len(loop.results) + k], assumed)
            _refuse_another_buffer_carried_back(arg, root, assumed)
            return root
        up = torchsim.view_root(v)
        if up == v:
            return v
        v = up


def _refuse_another_buffer_carried_back(arg, root, assumed):
    """A for loop's iter_arg that starts as one buffer is yielded back as that buffer, or the TOG
    would key every trip by the first one's."""
    block = arg.owner
    loop = getattr(block, "owner", None)
    loop = getattr(loop, "operation", loop)
    if loop is None or loop.name not in _FOR_LOOPS or arg.arg_number == 0:
        return
    term = block.operations[len(block.operations) - 1].operation
    k = arg.arg_number - 1
    if k >= len(term.operands) or not _is_memref(term.operands[k]):
        return
    back = _root(term.operands[k], tuple(assumed) + ((arg, root),))
    if back != root:
        raise RuntimeError(f"dep_analysis: {loop.name} carries a scratchpad buffer in as one "
                           f"buffer and yields another back; the TOG keys a buffer per value")


def _in_function_signature(v):
    if not isinstance(v, ir.BlockArgument):
        return False
    op = getattr(v.owner, "owner", None)
    op = getattr(op, "operation", op)
    return op is not None and op.name == "func.func"


def buffer_of(memref_val):
    """The buffer `memref_val` names, as the TOG keys it: a global's symbol, or the value an op
    allocates it as; None for one of no buffer (DRAM, a function argument). A scratchpad value
    whose buffer cannot be found is refused, never keyed None."""
    root = _root(memref_val)
    if isinstance(root, ir.OpResult) and root.owner.operation.name == "memref.get_global":
        return ir.FlatSymbolRefAttr(root.owner.operation.attributes["name"]).value
    if is_a_scratchpad_buffer(root):
        return root
    if in_scratchpad(memref_val.type) and not _in_function_signature(root):
        raise RuntimeError(f"dep_analysis: a scratchpad value of {memref_val.type} reaches no "
                           f"buffer: its views and carried inits stop at a value no op allocates")
    return None


def buffer_order(pos):
    """A sort key over buffer_of's keys: CORE and globals by name, then allocated buffers in the
    order `pos` (op -> walk position) puts the ops allocating them."""
    at = {getattr(o, "operation", o): k for o, k in pos.items()}

    def key(b):
        return (0, b, 0) if isinstance(b, str) else (1, "", at.get(b.owner.operation, 1 << 30))
    return key


def _label(b, pos):
    """A buffer_of key as text for a log: a global's symbol, or where its allocation stands."""
    if isinstance(b, str):
        return b
    return f"alloc@{buffer_order(pos)(b)[2]}"


def _skips(op, mrefs):
    """Does `op` only end the lives of the buffers it takes (ends_lifetime), touching no data?"""
    return all(ends_lifetime(op, v) for v in mrefs)


_LOAD_OPS = {"vector.transfer_read", "affine.vector_load", "vector.load",
             "memref.load", "affine.load", "torchsim.strided_load"}
_STORE_OPS = {"vector.transfer_write", "affine.vector_store", "vector.store",
              "memref.store", "affine.store", "torchsim.strided_store"}
_IGNORE_OPS = {"memref.extract_aligned_pointer_as_index",
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
        b = buffer_of(v)
        if b is not None:
            reads.setdefault(b)
    def wr(v):
        b = buffer_of(v)
        if b is not None:
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
        elif name in _IGNORE_OPS or _skips(op, mrefs):
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
    return buffer_of(val)


def compute_buffers(cn):
    """(read_buffers, write_buffers) for one segment: the scratchpad globals its ops touch, and
    CORE, which orders it after the core's previous segment."""
    reads, writes = _rw_buffers_of_compute(cn)
    return reads + [CORE], writes + [CORE]


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
        creads, cwrites = compute_buffers(cn)
        nodes.append({
            "kind": COMPUTE_TYPE_NAME[cn.compute_type],
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
    order = buffer_order(pos)
    for i, n in enumerate(nodes):
        for b in sorted(n["reads"], key=order):
            if b in last_writer:
                edges.append((i, last_writer[b], f"reads {_label(b, pos)}"))
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
        pos = {}
        for k, o in enumerate(_bs.walk_ops(module.body)):
            pos[o.operation] = k
    print("=== nodes ===")
    for i, n in enumerate(nodes):
        r = ",".join(_label(b, pos) for b in sorted(n["reads"], key=buffer_order(pos))) or "-"
        w = ",".join(_label(b, pos) for b in sorted(n["writes"], key=buffer_order(pos))) or "-"
        print(f"  #{i:<2} {n['kind']:<16} reads[{r}] writes[{w}]")
    print("=== edges (consumer -> producer) ===")
    for c, p, why in edges:
        print(f"  #{c} ({nodes[c]['kind']}) -> #{p} ({nodes[p]['kind']})   [{why}]")


if __name__ == "__main__":
    _main()
