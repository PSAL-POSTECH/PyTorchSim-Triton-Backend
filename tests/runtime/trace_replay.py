"""Run the trace producer the compiler emitted, and check what it says.

Building a `.so` proves the C++ compiles; it says nothing about whether the
trace describes the kernel that was compiled. This dlopens it, records every
`togsim_*` call, and holds the result against kernel.json and the post-vcix IR.
"""
import itertools
import json
import os
import subprocess
import tempfile

from pytorchsim_triton_compiler.contract.kernel_object import ITEMSIZE, PROGRAM_KEY_SHIFT

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "trace_replay.cpp")

#: What one element of each manifest dtype costs, for the `elem_bits` a dma
#: carries. The contract states it in bytes; absent means "do not check".
ELEM_BITS = {k: v * 8 for k, v in ITEMSIZE.items()}

#: Enumerating a transfer's element footprint is how coverage is checked; past
#: this many elements it is not worth the seconds, and the arg is left unjudged.
FOOTPRINT_CAP = 4 << 20



def binary():
    """The recorder, compiled on first use and cached outside the repo.

    Keyed by the mtime of the source AND of this file, so changing either the
    recorder or the flags below rebuilds it; written under a unique name then
    renamed, so two sweeps racing here cannot hand one a half-written file.
    """
    root = os.path.join(tempfile.gettempdir(), "torchsim-trace-replay")
    exe = os.path.join(root, "%d-%d" % (os.path.getmtime(SOURCE),
                                        os.path.getmtime(__file__)))
    if os.path.isfile(exe):
        return exe
    os.makedirs(root, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=root)
    os.close(fd)
    # -rdynamic, WITHOUT WHICH NOTHING LOADS: the producer leaves every togsim_*
    # undefined for its host to resolve, and a host that does not export its own
    # dynamic symbols cannot be that host.
    proc = subprocess.run(["g++", "-std=gnu++17", "-O1", "-rdynamic",
                           "-o", tmp, SOURCE, "-ldl"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        os.unlink(tmp)
        raise RuntimeError("could not build the trace recorder:\n" + proc.stderr)
    os.replace(tmp, exe)
    return exe


def replay(workdir, shape_args, timeout=600):
    """Every callback the producer makes, in order, one dict each."""
    so = os.path.join(workdir, "trace.so")
    proc = subprocess.run([binary(), so] + [str(s) for s in shape_args],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"the trace producer exited {proc.returncode}: "
                           f"{proc.stderr.strip()[:400]}")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def ir_transfers(workdir):
    """Every torchsim.transfer in the post-vcix IR, as the tuple a dma should be.

    THE IR IS WHAT THE ELF WAS BUILT FROM, so a trace naming other transfers is
    describing a different kernel than the one that ran. `masked` rides along
    because the ABI has no field for it -- see check().

    READ THROUGH THE BINDINGS, NOT A REGEX. This parsed the text once, and all
    three of its patterns aged out of the IR at once -- the op name, the spad
    memref's layout and the dense spelling of dram_stride -- so it reported an
    empty IR and every dma the trace emitted looked invented. The transfer's ODS
    owns the operand and attribute names; asking it cannot go stale.
    """
    path = os.path.join(workdir, "05-custom.mlir")
    if not os.path.isfile(path):
        return None
    import mlir.ir as ir
    from pytorchsim_triton_compiler.dialects import torchsim
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import (
        elem_bits, int_array, walk)
    ctx = ir.Context()
    # The post-vcix IR may carry ops of a dialect this LLVM does not register
    # -- the same flag pipeline.py parses it with.
    ctx.allow_unregistered_dialects = True
    out = []
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(open(path).read())
        for op in walk(module.operation):
            if op.name != torchsim.TransferOp.OPERATION_NAME:
                continue
            attrs = op.attributes
            if "dma_kind" not in attrs or "dram_arg" not in attrs:
                continue
            tx = op.operation.opview
            spad = ir.MemRefType(tx.sram.type)
            stride = tx.dram_stride
            out.append({
                "dir": 0 if ir.StringAttr(attrs["dma_kind"]).value == "MVIN"
                       else 1,
                "arg": ir.IntegerAttr(attrs["dram_arg"]).value,
                "dims": (tuple(int_array(tx.subtile_size)) if tx.subtile_size is not None
                         else tuple(spad.shape)),
                "strides": tuple(int_array(stride)) if stride else (),
                "elem_bits": elem_bits(spad.element_type),
                "masked": "masked_axes" in attrs,
                "indirect": tx.index is not None,
                "cut": tx.cut is not None,
            })
    return out


def ir_scalar_dram(workdir):
    """Manifest arg -> the directions a SCALAR access reaches it in, outside any
    dma. A memref.load/store straight to a DRAM argument is real traffic that
    togsim_dma never carries, so an arg the trace leaves untouched may be moved
    after all -- by an access the trace cannot express."""
    path = os.path.join(workdir, "05-custom.mlir")
    if not os.path.isfile(path):
        return {}
    import mlir.ir as ir
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk

    def dram(v):
        try:
            return ir.MemRefType(v.type).memory_space is None
        except Exception:
            return False

    def arg_of(v):
        while not isinstance(v, ir.BlockArgument):
            op = v.owner if isinstance(v.owner, ir.Operation) else \
                 getattr(v.owner, "operation", None)
            if op is None or not op.operands:
                return None
            v = op.operands[0]
        return v.arg_number

    out = {}
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(open(path).read())
        for op in walk(module.operation):
            if op.name in ("memref.store", "affine.store"):
                v, d = op.operands[1], 1
            elif op.name in ("memref.load", "affine.load"):
                v, d = op.operands[0], 0
            else:
                continue
            if not dram(v):
                continue
            i = arg_of(v)
            if i is not None:
                out.setdefault(i, set()).add(d)
    return out


def _footprint(event):
    """Element indices the event touches, or None if it is not worth listing."""
    dims = event["dims"] or [1]
    strides = event["strides"] or [1] * len(dims)
    n = 1
    for d in dims:
        n *= d
    if n > FOOTPRINT_CAP:
        return None
    base = event["offset"]
    return {base + sum(p * s for p, s in zip(pos, strides))
            for pos in itertools.product(*[range(d) for d in dims])}


def _key(t):
    """A transfer's identity, from the IR or from a dma, in one spelling.

    TUPLES ON BOTH SIDES: the IR side builds them, the trace side comes back
    from JSON as lists, and a list cannot go in the set that compares them.
    """
    return (t["dir"], t["arg"], tuple(t["dims"]), tuple(t["strides"]),
            t["elem_bits"], bool(t["indirect"]))


def check(workdir):
    """(problems, notes) for the trace in `workdir`. Empty problems is a pass.

    A NOTE IS NOT A NEAR-MISS: it names an approximation the trace ABI cannot
    express (a clamped mask, a lapped cut), which is why the footprint checks
    below stand down for the args that carry one -- and for a gather's.
    """
    manifest = json.load(open(os.path.join(workdir, "kernel.json")))
    types = json.load(open(os.path.join(workdir, "tile_types.json")))
    args = manifest["args"]
    axes = manifest["abi"]["parallel_axes"]
    grid = manifest["grid"][:len(axes)] or [1]
    bad, notes = [], []

    events = replay(workdir, grid)
    transfers = ir_transfers(workdir) or []

    # 1. one dispatch per program, and the ids enumerate the grid
    want = [list(reversed(w))
            for w in itertools.product(*[range(g) for g in reversed(grid)])]
    got = [e["iv"] for e in events if e["op"] == "dispatch"]
    if len(got) != len(want):
        bad.append(f"{len(got)} work items dispatched, the grid {grid} says "
                   f"{len(want)}")
    elif sorted(got) != sorted(want):
        bad.append(f"the work-item ids do not enumerate the grid {grid}: "
                   f"{got[:4]}...")
    for e in (e for e in events if e["op"] == "dispatch"):
        key = sum(i << PROGRAM_KEY_SHIFT[a] for i, a in zip(e["iv"], axes))
        if e["key"] != key:
            bad.append(f"work item {e['iv']} is dispatched under key {e['key']}, and the "
                       f"wrapper names that program {key}: its index dumps would not be found")
            break

    # 2. the trace names the same transfers the IR does
    in_ir = {_key(t) for t in transfers}
    in_trace = {_key(e) for e in events if e["op"] == "dma"}
    for t in sorted(in_ir - in_trace):
        bad.append(f"the IR has a transfer the trace never emits: dir={t[0]} "
                   f"arg={t[1]} dims={list(t[2])} strides={list(t[3])} indirect={t[5]}")
    for t in sorted(in_trace - in_ir):
        bad.append(f"the trace emits a dma the IR does not have: dir={t[0]} "
                   f"arg={t[1]} dims={list(t[2])} strides={list(t[3])} indirect={t[5]}")

    # WHICH ARGS CANNOT BE JUDGED, and why. A masked transfer's descriptor is
    # clamped by dim_low/dim_high, which never reaches togsim_dma; a gather's
    # addresses come from the functional run's index dumps, which this replay
    # does not have. Either way the footprint here is not the kernel's.
    approximate = set()
    for t in transfers:
        if t["masked"]:
            approximate.add(t["arg"])
            notes.append(f"arg {t['arg']}: the transfer is masked and the trace "
                         f"carries the unclamped extent {list(t['dims'])}")
        if t["indirect"]:
            approximate.add(t["arg"])
        if t["cut"]:
            approximate.add(t["arg"])
            notes.append(f"arg {t['arg']}: the transfer is cut into laps and "
                         f"the trace carries the whole box {list(t['dims'])}")

    numel = []
    for a in args:
        n = 1
        for d in a["shape"]:
            n *= d
        numel.append(n)

    # 3. every dma names a tensor the manifest has, in the manifest's own dtype
    #: UNLESS THE IR SAYS OTHERWISE. `tl.bitcast` reads a buffer at another
    #: element width and the transfer states it; check 2 already holds the trace
    #: to the IR's elem_bits, so disagreeing with the MANIFEST here is the
    #: kernel's doing and not the trace's.
    ir_bits = {(t["arg"], t["elem_bits"]) for t in transfers}
    # A ROUND TRIP THROUGH DRAM NAMES NO ARGUMENT. lower_copies mints a
    # global for one and states torchsim.DRAM_ARG_ROUND_TRIP, so the manifest has
    # nothing to resolve it against and the checks below stand down for it --
    # any OTHER out-of-range index is still the bug this check is here for.
    from pytorchsim_triton_compiler.dialects.torchsim import (DRAM_ARG_POINTER_TILE,
                                                              DRAM_ARG_ROUND_TRIP)
    touched, unjudged = {i: set() for i in range(len(args))}, set()
    scratch = pointers = 0
    for e in events:
        if e["op"] != "dma":
            continue
        i = e["arg"]
        if i == DRAM_ARG_ROUND_TRIP:
            scratch += 1
            continue
        if i == DRAM_ARG_POINTER_TILE:
            pointers += 1
            continue
        if not 0 <= i < len(args):
            bad.append(f"a dma names arg {i}; the manifest has {len(args)}, so "
                       f"the simulator resolves it to address 0")
            continue
        want_bits = ELEM_BITS.get(args[i]["dtype"])
        if want_bits and e["elem_bits"] != want_bits:
            (notes if (i, e["elem_bits"]) in ir_bits else bad).append(
                f"arg {i} ({args[i]['name']}) is {args[i]['dtype']} and "
                f"the dma says elem_bits={e['elem_bits']}")
        if args[i]["role"] == "in" and e["dir"] == 1:
            bad.append(f"arg {i} ({args[i]['name']}) is `in` and a dma stores to it")
        if i in approximate:
            continue
        fp = _footprint(e)
        if fp is None:
            unjudged.add(i)
            continue
        past = [x for x in fp if x >= numel[i] or x < 0]
        if past:
            bad.append(f"arg {i} ({args[i]['name']}) has {numel[i]} elements and "
                       f"a dma reaches {min(past)}..{max(past)}")
        touched[i] |= fp

    if scratch:
        notes.append(f"{scratch} dma(s) move a compiler-minted DRAM buffer, "
                     f"which is no argument of the kernel")
    if pointers:
        notes.append(f"{pointers} dma(s) go through a tile of pointers, addressed "
                     f"over all of memory rather than from one argument")

    # 4. what the launch never moves, the simulator never sees. A NOTE AND NOT
    # A PROBLEM, for two reasons. NOT EVERY ARG IS FULLY MOVED, and that is the
    # kernel's business: an oversized output written in part is a correct kernel
    # (ops_shape stores 40 of 64), and check 2 already proves the trace names
    # the transfers the IR has. AND SCALAR TRAFFIC IS NOT MODELLED AT ALL --
    # a memref.load/store straight to DRAM is outside what this backend times,
    # by decision, so an arg reached only that way is named, not failed.
    scalar = ir_scalar_dram(workdir)
    for i, a in enumerate(args):
        if i in approximate or i in unjudged:
            continue
        why = (" -- a scalar DRAM access reaches it, and scalar traffic is not "
               "modelled") if i in scalar else ""
        if not touched[i]:
            notes.append(f"arg {i} ({a['name']}) is never moved by any dma, so "
                         f"the simulator sees no traffic for it{why}")
        elif a["role"] == "out":
            missing = numel[i] - len(touched[i] & set(range(numel[i])))
            if missing:
                notes.append(f"arg {i} ({a['name']}) is `out` and {missing} of "
                             f"its {numel[i]} elements are never stored{why}")

    # 5. the tiles agree with the table that was written beside them
    table = types["compute_types"]
    seen = set()
    for e in events:
        if e["op"] != "compute":
            continue
        t = e["tile"]
        seen.add(t)
        if t >= len(table):
            bad.append(f"a compute names tile {t}; tile_types.json has {len(table)}")
        elif table[t] != e["type"]:
            bad.append(f"tile {t} is type {e['type']} in the trace and "
                       f"{table[t]} in tile_types.json")
        if not e["read"] and not e["write"]:
            notes.append(f"tile {t} reads nothing and writes nothing")
    for t in range(len(table)):
        if t not in seen:
            bad.append(f"tile {t} is in tile_types.json and never emitted")

    # 6. dataflow inside a work item. A NODE'S OWN WRITE SATISFIES ITS OWN READ:
    # the TOG fuses a zero-init with the accumulate that follows, so one tile
    # both reads and writes the accumulator and that is not a dangling read.
    for item in _work_items(events):
        written = set()
        for e in item:
            written |= set(e.get("write", []))
        for e in item:
            for b in e.get("read", []):
                if b not in written:
                    bad.append(f"tile {e.get('tile', e.get('arg'))} reads buffer "
                               f"{b}, which nothing in the work item writes")
    return bad, sorted(set(notes))


def _work_items(events):
    """The event stream cut at each dispatch, one list per work item."""
    item, out = [], []
    for e in events:
        if e["op"] == "dispatch" and item:
            out.append(item)
            item = []
        item.append(e)
    if item:
        out.append(item)
    return out
