"""Record the TOG the compiler emitted, and check what it says.

Building a `.so` proves the C++ compiles; it says nothing about whether the
TOG describes the kernel that was compiled. This dlopens it, records every
`togsim_*` call, and holds the result against kernel.json and the post-vcix IR.
"""
import itertools
import json
import math
import os
import subprocess
import tempfile

import numpy as np

from pytorchsim_triton_compiler import toolchain
from pytorchsim_triton_compiler.contract import kernel_object
from pytorchsim_triton_compiler.contract.kernel_object import (CYCLE_CALLS, ITEMSIZE, PROGRAM_KEY_SHIFT,
                                                               RUNTIME_ATTRIBUTE_DIR)

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "tog_record.cpp")

#: What one element of each manifest dtype costs, for the `elem_bits` a dma
#: carries. The contract states it in bytes; absent means "do not check".
ELEM_BITS = {k: v * 8 for k, v in ITEMSIZE.items()}

#: Enumerating a transfer's element footprint is how coverage is checked; past this many elements
#: the arg is left unjudged. Set when a footprint cost seconds, and kept so no verdict moves.
FOOTPRINT_CAP = 4 << 20



def binary():
    """The recorder, compiled on first use and cached outside the repo.

    Keyed by the mtime of the source AND of this file, so changing either the
    recorder or the flags below rebuilds it; written under a unique name then
    renamed, so two sweeps racing here cannot hand one a half-written file.
    """
    root = os.path.join(tempfile.gettempdir(), "torchsim-tog-record")
    header = os.path.join(toolchain.TOGSIM_INCLUDE, "togsim_runtime.h")
    exe = os.path.join(root, "%d-%d-%d" % (os.path.getmtime(SOURCE), os.path.getmtime(__file__),
                                           os.path.getmtime(header)))
    if os.path.isfile(exe):
        return exe
    os.makedirs(root, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=root)
    os.close(fd)
    # -rdynamic, WITHOUT WHICH NOTHING LOADS: the TOG leaves every togsim_*
    # undefined for its host to resolve, and a host that does not export its own
    # dynamic symbols cannot be that host.
    proc = subprocess.run(["g++", "-std=gnu++20", "-O1", "-rdynamic", "-I", toolchain.TOGSIM_INCLUDE,
                           "-o", tmp, SOURCE, "-ldl"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        os.unlink(tmp)
        raise RuntimeError("could not build the TOG recorder:\n" + proc.stderr)
    os.replace(tmp, exe)
    return exe


def attributes_of(runtime_dir):
    """Where a launch's runtime attributes are, under its runtime dir: what the recorder reads."""
    return os.path.join(runtime_dir, RUNTIME_ATTRIBUTE_DIR)


def record(workdir, shape_args, timeout=600, attributes=None, so=None):
    """Every callback the TOG makes (tog_so, or the TOG `so`), in order, one dict each, reading the runtime
    attributes under `attributes` (the workdir's runtime/ when not given) when the run left them."""
    so = so or kernel_object.artifact(workdir, "tog_so")
    attributes = attributes or attributes_of(os.path.join(workdir, "runtime"))
    flag = ["--attributes", attributes] if os.path.isdir(attributes) else []
    proc = subprocess.run([binary()] + flag + [so] + [str(s) for s in shape_args],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"the TOG recorder exited {proc.returncode}: "
                           f"{proc.stderr.strip()[:400]}")
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def ir_transfers(workdir):
    """Every pytorchsim.transfer in the post-vcix IR, as the tuple a dma should be.

    THE IR IS WHAT THE ELF WAS BUILT FROM, so a TOG naming other transfers is
    describing a different kernel than the one that ran. `masked` rides along
    because the ABI has no field for it -- see check().

    READ THROUGH THE BINDINGS, NOT A REGEX. This parsed the text once, and all
    three of its patterns aged out of the IR at once -- the op name, the spad
    memref's layout and the dense spelling of dram_stride -- so it reported an
    empty IR and every dma the TOG emitted looked invented. The transfer's ODS
    owns the operand and attribute names; asking it cannot go stale.
    """
    path = os.path.join(workdir, "09-custom.mlir")
    if not os.path.isfile(path):
        return None
    import mlir.ir as ir
    from pytorchsim_triton_compiler import machine
    from pytorchsim_triton_compiler.dialects import pytorchsim
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import elem_bits, walk
    from pytorchsim_triton_compiler.lowering.passes.lower import dma_lowering
    from pytorchsim_triton_compiler.lowering.tog.passes.build_skeleton import run_time_sized_axes
    walked = dma_lowering(machine.target()).walk
    ctx = ir.Context()
    # The post-vcix IR may carry ops of a dialect this LLVM does not register
    # -- the same flag pipeline.py parses it with.
    ctx.allow_unregistered_dialects = True
    out = []
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(open(path).read())
        for op in walk(module.operation):
            if op.name != pytorchsim.TransferOp.OPERATION_NAME:
                continue
            attrs = op.attributes
            if "direction" not in attrs or "dram_arg" not in attrs:
                continue
            tx = op.operation.opview
            spad = ir.MemRefType(tx.sram.type)
            dims, strides, lead = walked(op, machine.VECTORLANE_SIZE)
            out.append({
                "dir": 0 if pytorchsim.is_load(op) else 1,
                "arg": ir.IntegerAttr(attrs["dram_arg"]).value,
                "dims": tuple(dims),
                "strides": tuple(strides),
                "elem_bits": elem_bits(spad.element_type),
                "masked": "masked_axes" in attrs,
                "clamped_axes": tuple(lead + a for a in run_time_sized_axes(op)),
                "indirect": tx.index is not None,
            })
    return out


def descriptor_counts(workdir):
    """The element counts the DMA descriptors of 10-lowered.mlir walk, prod(high - low) over their dims,
    decoded through the bindings in the lower-dma provider's own word layout; None without the file."""
    path = os.path.join(workdir, "10-lowered.mlir")
    if not os.path.isfile(path):
        return None
    import mlir.ir as ir
    from pytorchsim_triton_compiler import machine
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk
    from pytorchsim_triton_compiler.lowering.passes.lower import dma_lowering
    dma = dma_lowering(machine.target())
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    out = set()
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(open(path).read())
        for op in walk(module.operation):
            if (op.name != "llvm.mlir.global"
                    or not ir.StringAttr(op.attributes["sym_name"]).value.startswith("dma_desc_")):
                continue
            slots = ir.DenseIntElementsAttr(op.attributes["value"])
            width = ir.IntegerType(ir.ShapedType(slots.type).element_type).width
            word = [(int(slots[i]) >> (32 * h)) & 0xFFFFFFFF
                    for i in range(len(slots)) for h in range(width // 32)]
            out.add(math.prod(word[dma.W_DIM_HIGH + d] - word[dma.W_DIM_LOW + d]
                              for d in range(dma.DESCRIPTOR_DIMS)))
    return out


def ir_branch_tiles(workdir):
    """The tile ids the post-vcix IR stamps on an op inside a branch's region: a tile the TOG emits
    only when the run's data or ids take that arm."""
    path = os.path.join(workdir, "09-custom.mlir")
    if not os.path.isfile(path):
        return set()
    import mlir.ir as ir
    from pytorchsim_triton_compiler.dialects import pytorchsim
    from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk
    from pytorchsim_triton_compiler.lowering.tog.passes.build_tog import TILE_ID
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    out = set()
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(open(path).read())
        for op in walk(module.operation):
            if TILE_ID not in op.attributes:
                continue
            p = op.parent
            while p is not None and pytorchsim.branch_controls(p) is None:
                p = p.parent
            if p is not None:
                out.add(ir.IntegerAttr(op.attributes[TILE_ID]).value)
    return out


def ir_scalar_dram(workdir):
    """Manifest arg -> the directions a SCALAR access reaches it in, outside any
    dma. A memref.load/store straight to a DRAM argument is real traffic that
    togsim_dma never carries, so an arg the TOG leaves untouched may be moved
    after all -- by an access the TOG cannot express."""
    path = os.path.join(workdir, "09-custom.mlir")
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
    """Element indices the event touches, as an array that may repeat one, or None past the cap.
    Dims pair with strides as zip() pairs them; a zero extent touches nothing. An index past int64
    (offset is a uint64) stays a Python int, in an object array, so no bound wraps."""
    dims = event["dims"] or [1]
    strides = event["strides"] or [1] * len(dims)
    if math.prod(dims) > FOOTPRINT_CAP:
        return None
    if any(d <= 0 for d in dims):
        return np.empty(0, np.int64)
    pairs = list(zip(dims, strides))
    base = event["offset"]
    wide = abs(base) + sum(abs((d - 1) * s) for d, s in pairs) >= 1 << 63
    dtype = object if wide else np.int64
    idx = np.array([base], dtype)
    for d, s in pairs:
        idx = (idx[:, None] + np.arange(d, dtype=dtype) * s).ravel()
    return idx


def _key(t):
    """A transfer's identity, from the IR or from a dma, in one spelling.

    TUPLES ON BOTH SIDES: the IR side builds them, the TOG side comes back
    from JSON as lists, and a list cannot go in the set that compares them.
    """
    return (t["dir"], t["arg"], tuple(t["dims"]), tuple(t["strides"]),
            t["elem_bits"], bool(t["indirect"] if "indirect" in t else "offsets" in t))


def _handed_extents(workdir, manifest, types, axes, grid, events):
    """Problems with what a work-item is handed after its ids: the extents of the axes
    tile_types.json names, then the values of the scalars it names, the same in every dispatch, on
    `grid` and on a grid one program longer on every axis -- the shape the TOG is run with, not the
    one it was compiled for."""
    names = types.get("dispatch_extents", [])
    scalars = types.get("dispatch_scalars", [])
    value = {s["name"]: int(kernel_object.scalar_value(manifest, s))
             for s in kernel_object.launch_scalars(manifest.get("scalars", []))}
    keyed = any(e["op"] == "attributes" and e["ranks"] for e in events)
    bad = []
    for shape in [grid] + ([[g + 1 for g in grid]] if names and not keyed else []):
        evs = events if shape == grid else record(workdir, kernel_object.shape_args(manifest, grid=shape))
        want = [shape[axes.index(a)] for a in names] + [value[n] for n in scalars]
        got = [e["iv"][len(grid):] for e in evs if e["op"] == "dispatch"]
        odd = [t for t in got if t != want]
        if odd:
            bad.append(f"run on the grid {shape}, a work-item is handed {odd[0]} after its ids; "
                       f"tile_types.json says the extents of {names} and the scalars {scalars}, "
                       f"{want}")
        if len(got) != math.prod(shape):
            bad.append(f"run on the grid {shape}, {len(got)} work items are dispatched")
    return bad


def check(workdir):
    """(problems, notes) for the TOG in `workdir`. Empty problems is a pass.

    A NOTE IS NOT A NEAR-MISS: it names an approximation the TOG ABI cannot
    express (a clamped mask, a lapped cut), which is why the footprint checks
    below stand down for the args that carry one -- and for a gather's.
    """
    manifest = json.load(open(os.path.join(workdir, "kernel.json")))
    types = json.load(open(os.path.join(workdir, "tile_types.json")))
    args = manifest["args"]
    axes = manifest["abi"]["parallel_axes"]
    grid = kernel_object.launch_shape(manifest)
    bad, notes = [], []

    events = record(workdir, kernel_object.shape_args(manifest))
    copies = kernel_object.copy_nodes(manifest)
    held_args = {c["arg"] for c in copies}
    held_buffers = {c["held_buffer"] for c in copies}
    bad += [p for c in copies for p in copy_node_problems(workdir, c, args)]
    every = ir_transfers(workdir) or []
    bad += [f"the copy node holds arg {a}, and the IR moves it by {n} transfers, not one load"
            for a in held_args for n in [sum(t["arg"] == a for t in every)]
            if n != 1 or any(t["arg"] == a and t["dir"] != 0 for t in every)]
    transfers = [t for t in every if t["arg"] not in held_args]
    bad += [f"the TOG loads arg {e['arg']}, which its copy node holds" for e in events
            if e["op"] == "dma" and e["dir"] == 0 and e["arg"] in held_args]

    # 1. one dispatch per program, and the ids enumerate the grid
    want = [list(reversed(w))
            for w in itertools.product(*[range(g) for g in reversed(grid)])]
    got = [e["iv"][:len(grid)] for e in events if e["op"] == "dispatch"]
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
                       f"wrapper names that program {key}: its runtime attribute records would not be found")
            break
    bad += _handed_extents(workdir, manifest, types, axes, grid, events)

    # 2. the TOG names the same transfers the IR does. A clamped axis may be sized at run
    # time, so there a dma matches a transfer whose extent is at least its own.
    def matches(t, k):
        d, a, dims, strides, bits, indirect = k
        if ((d, a, strides, bits, indirect) != _key(t)[:2] + _key(t)[3:]
                or len(dims) != len(t["dims"])):
            return False
        return all(1 <= x <= y if i in t["clamped_axes"] else x == y
                   for i, (x, y) in enumerate(zip(dims, t["dims"])))

    in_tog = {_key(e) for e in events if e["op"] == "dma"}
    seen = set()
    for t in transfers:
        hits = {k for k in in_tog if matches(t, k)}
        seen |= hits
        if not hits:
            k = _key(t)
            bad.append(f"the IR has a transfer the TOG never emits: dir={k[0]} "
                       f"arg={k[1]} dims={list(k[2])} strides={list(k[3])} indirect={k[5]}")
    for t in sorted(in_tog - seen):
        bad.append(f"the TOG emits a dma the IR does not have: dir={t[0]} "
                   f"arg={t[1]} dims={list(t[2])} strides={list(t[3])} indirect={t[5]}")

    walks = descriptor_counts(workdir)
    sized = {k for k in in_tog if any(matches(t, k) and t["clamped_axes"] for t in transfers)}
    for k in sorted(in_tog - sized) if walks is not None else ():
        if math.prod(k[2]) not in walks:
            bad.append(f"the TOG states {math.prod(k[2])} elements for dir={k[0]} "
                       f"arg={k[1]} dims={list(k[2])}, and no descriptor in "
                       f"10-lowered.mlir walks that many: {sorted(walks)}")

    # WHICH ARGS CANNOT BE JUDGED, and why. A masked transfer's descriptor is
    # clamped by dim_low/dim_high, which never reaches togsim_dma; a gather's
    # addresses rest on the index snapshot the run recorded, which this check
    # does not follow. Either way the footprint here is not the kernel's.
    approximate = set()
    for t in transfers:
        if t["masked"]:
            approximate.add(t["arg"])
            notes.append(f"arg {t['arg']}: the transfer is masked; the TOG sizes axes "
                         f"{list(t['clamped_axes'])} of {list(t['dims'])} at run time when it "
                         f"can compute them" if t["clamped_axes"] else
                         f"arg {t['arg']}: the transfer is masked and the TOG "
                         f"carries the unclamped extent {list(t['dims'])}")
        if t["indirect"]:
            approximate.add(t["arg"])

    numel = []
    for a in args:
        n = 1
        for d in a["shape"]:
            n *= d
        numel.append(n)

    # 3. every dma names a tensor the manifest has, in the manifest's own dtype
    #: UNLESS THE IR SAYS OTHERWISE. `tl.bitcast` reads a buffer at another
    #: element width and the transfer states it; check 2 already holds the TOG
    #: to the IR's elem_bits, so disagreeing with the MANIFEST here is the
    #: kernel's doing and not the TOG's.
    ir_bits = {(t["arg"], t["elem_bits"]) for t in transfers}
    from pytorchsim_triton_compiler.dialects.pytorchsim import DRAM_ARG_POINTER_TILE
    moved, covered, unjudged = {i: False for i in range(len(args))}, {}, set()
    pointers = 0
    for e in events:
        if e["op"] != "dma":
            continue
        i = e["arg"]
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
        out = (fp < 0) | (fp >= numel[i])
        if out.any():
            past = fp[out]
            bad.append(f"arg {i} ({args[i]['name']}) has {numel[i]} elements and "
                       f"a dma reaches {int(past.min())}..{int(past.max())}")
        moved[i] |= fp.size > 0
        inside = fp[~out]
        if args[i]["role"] == "out" and inside.size:
            if i not in covered:
                covered[i] = np.zeros(numel[i], bool)
            covered[i][inside.astype(np.int64)] = True

    if pointers:
        notes.append(f"{pointers} dma(s) go through a tile of pointers, addressed "
                     f"over all of memory rather than from one argument")

    # 4. what the launch never moves, the simulator never sees. A NOTE AND NOT
    # A PROBLEM, for two reasons. NOT EVERY ARG IS FULLY MOVED, and that is the
    # kernel's business: an oversized output written in part is a correct kernel
    # (ops_shape stores 40 of 64), and check 2 already proves the TOG names
    # the transfers the IR has. AND SCALAR TRAFFIC IS NOT MODELLED AT ALL --
    # a memref.load/store straight to DRAM is outside what this backend times,
    # by decision, so an arg reached only that way is named, not failed.
    scalar = ir_scalar_dram(workdir)
    for i, a in enumerate(args):
        if i in approximate or i in unjudged:
            continue
        why = (" -- a scalar DRAM access reaches it, and scalar traffic is not "
               "modelled") if i in scalar else ""
        if not moved[i] and i not in held_args:
            notes.append(f"arg {i} ({a['name']}) is never moved by any dma, so "
                         f"the simulator sees no traffic for it{why}")
        elif a["role"] == "out":
            missing = numel[i] - (int(np.count_nonzero(covered[i])) if i in covered else 0)
            if missing:
                notes.append(f"arg {i} ({a['name']}) is `out` and {missing} of "
                             f"its {numel[i]} elements are never stored{why}")

    # 5. the tiles agree with the table that was written beside them: a compute names row
    # t + n * c of the cycle table, call c of tile t, and every compute the last call
    table = types["compute_types"]
    seen = set()
    for e in events:
        if e["op"] != "compute":
            continue
        t, call = (e["tile"] % len(table), e["tile"] // len(table)) if table else (e["tile"], 0)
        seen.add(t)
        if e["tile"] >= len(table) * CYCLE_CALLS:
            bad.append(f"a compute names row {e['tile']}; tile_types.json has {len(table)} "
                       f"tile(s) of {CYCLE_CALLS} call(s)")
        elif call != CYCLE_CALLS - 1:
            bad.append(f"a compute names row {e['tile']}, call {call} of tile {t}; the TOG "
                       f"computes on call {CYCLE_CALLS - 1}")
        elif table[t] != e["type"]:
            bad.append(f"tile {t} is type {e['type']} in the TOG and "
                       f"{table[t]} in tile_types.json")
        if not e["read"] and not e["write"]:
            notes.append(f"tile {t} reads nothing and writes nothing")
    armed = ir_branch_tiles(workdir)
    for t in range(len(table)):
        if t not in seen and t in armed:
            notes.append(f"tile {t} sits in a branch arm this run never took")
        elif t not in seen:
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
                if b not in written and b not in held_buffers:
                    bad.append(f"tile {e.get('tile', e.get('arg'))} reads buffer "
                               f"{b}, which nothing in the work item writes")

    return bad, sorted(set(notes))


def copy_node_problems(workdir, node, args):
    """What is wrong with one copy node of the kernel object in `workdir`: it must be one work item whose one
    dma loads its argument whole, contiguously from element 0, and nothing else."""
    arg = node["arg"]
    if not 0 <= arg < len(args):
        return [f"a copy node holds arg {arg}; the manifest has {len(args)}"]
    events = record(workdir, [], so=os.path.join(workdir, node["tog_so"]))
    numel, want = math.prod(args[arg]["shape"]), ELEM_BITS.get(args[arg]["dtype"])
    dmas = [e for e in events if e["op"] == "dma"]
    bad = [f"the copy node of arg {arg} has {e['op']} {e.get('tile', '')}" for e in events
           if e["op"] not in ("dispatch", "dma", "attributes")]
    if sum(e["op"] == "dispatch" for e in events) != 1:
        bad.append(f"the copy node of arg {arg} dispatches {sum(e['op'] == 'dispatch' for e in events)} work items")
    whole = {"dir": 0, "arg": arg, "offset": 0, "dims": [numel], "strides": [1], "elem_bits": want}
    if len(dmas) != 1 or any(dmas[0].get(k) != v for k, v in whole.items()):
        bad.append(f"the copy node of arg {arg} moves {[{k: e.get(k) for k in whole} for e in dmas]}, "
                   f"not one load of its {numel} elements: {whole}")
    if node["bytes"] != numel * ITEMSIZE[args[arg]["dtype"]]:
        bad.append(f"the copy node of arg {arg} says {node['bytes']} B; the arg is {numel} {args[arg]['dtype']}")
    return bad


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
