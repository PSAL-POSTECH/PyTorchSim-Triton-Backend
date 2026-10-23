"""Spike, the functional device: the kernel object's ELF over the .raw files of its arguments, the grid
cut into program ranges across processes when no argument is read as well as written. The caller says
where Spike and pk are, how a process is spawned and how a failure is spelled; the rest is kernel.json."""
import concurrent.futures as cf
import math
import os
import shutil
import subprocess

from ..contract import kernel_object as ko

TRAP_HINT = {200: "INVALID_SPAD_ACCESS", 201: "STACK_OVERFLOW"}


class SpikeError(RuntimeError):
    """A Spike process, or the launch around it, failed."""


class NotSplittable(Exception):
    """Which program ran last decides the answer, so the split cannot give it."""


def run_process(cmd, cwd, log):
    """Spawn `cmd` in `cwd`; a nonzero exit raises SpikeError naming the trap (TRAP_HINT) and quoting
    the last 40 lines it printed."""
    log("[spike] $ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        said = (proc.stdout + proc.stderr).strip().splitlines()
        hint = TRAP_HINT.get(proc.returncode)
        raise SpikeError(f"spike exit code {proc.returncode}" + (f" ({hint})" if hint else "")
                         + ("\n" + "\n".join(said[-40:]) if said else ""))
    return proc


def name_trap(e):
    """`e`, a failed run's exception, with the trap its exit code means appended to its message."""
    for code, name in TRAP_HINT.items():
        if f"exit code {code}" in str(e):
            e.args = (f"{e.args[0]}  ({name})",)
    return e


def command(spike, pk, elf, manifest, runtime, raw_paths, options, scalar_values=None, worker=0):
    """Spike's argv for the whole grid under `options` (kernel_object.run_options): the .raw paths as
    basenames, since pk's argv is a fixed 2048 bytes and Spike runs with cwd=runtime; then the record
    file of `worker`, when the kernel records runtime attributes; then the scalars."""
    lo, hi = manifest["abi"]["kernel_addr"]
    return [spike, "--isa", options["spike_isa"], f"--extlib={options['model']}",
            "-m" + options["memory"], f"--machine-config={options['machine_config']}",
            f"--kernel-addr={lo:x}:{hi:x}", f"--base-path={runtime}", pk, elf,
            *(os.path.basename(raw_paths[n]) for n in ko.argv_order(manifest)),
            *ko.argv_records(manifest, worker), *ko.argv_tail(manifest, scalar_values)]


def programs(manifest):
    """How many programs the launch grid holds."""
    total = 1
    for g in ko.grid3(manifest["grid"]):
        total *= g
    return total


def jobs(manifest, total, log, want=None):
    """How many processes walk the grid: 1 when an inout argument lets one program read another's store,
    else `want` (the caller's override), else min(8, half the cores); never more than `total`."""
    if not ko.splittable(manifest):
        inout = [a["name"] for a in manifest["args"] if a["role"] == "inout"]
        log(f"[spike] one process: {', '.join(inout)} is read as well as "
            f"written, so a program may depend on another one's store")
        return 1
    if want:
        return max(1, min(int(want), total))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(8, cpus // 2, total))


def launch(elf, manifest, runtime, raw_paths, options, *, spike, pk, log, run=None,
           error=SpikeError, want_jobs=None, scalar_values=None):
    """Run the whole grid, leaving each written argument in its .raw and the runtime attributes it
    records under runtime/runtime_attribute/; `run(cmd, cwd)` spawns one process (run_process by
    default) and `error(message)` is the exception a failed merge or record raises."""
    run = run or (lambda cmd, cwd: run_process(cmd, cwd, log))
    os.makedirs(os.path.join(runtime, "dma_access"), exist_ok=True)
    cmd = command(spike, pk, elf, manifest, runtime, raw_paths, options, scalar_values)
    total = programs(manifest)
    n = jobs(manifest, total, log, want_jobs)
    workers = [0]
    if n > 1:
        workers = _split(manifest, raw_paths, cmd, runtime, total, n, log, run, error)
    else:
        start_records(manifest, runtime, workers)
        run(cmd, runtime)
    record_attributes(manifest, runtime, workers, log, error)
    return runtime


def start_records(manifest, runtime, workers):
    """An empty runtime/runtime_attribute/ holding an empty record file per worker, which the run
    fills: pk opens only a file that exists."""
    out = os.path.join(runtime, ko.RUNTIME_ATTRIBUTE_DIR)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    if manifest.get("runtime_attributes"):
        for w in workers:
            open(os.path.join(runtime, ko.record_file(w)), "wb").close()


def record_attributes(manifest, runtime, workers, log, error=SpikeError):
    """The run's record files as the files a TOG reads, under runtime/runtime_attribute/, or nothing
    for a kernel that declares no runtime attributes."""
    if not manifest.get("runtime_attributes"):
        return
    try:
        keys, nbytes = write_runtime_attributes(manifest, runtime, workers)
    except ko.ObjectError as e:
        raise error(str(e)) from None
    log(f"[spike] runtime attributes of {len(keys)} program(s), {nbytes} record bytes, in "
        f"{os.path.join(runtime, ko.RUNTIME_ATTRIBUTE_DIR)}")


def read_records(manifest, path):
    """The records of the record file at `path`, as (key, id, bytes) in file order; ObjectError when
    it is missing or its header is not this kernel's."""
    import struct
    got = manifest["runtime_attributes"]
    if not os.path.isfile(path):
        raise ko.ObjectError(f"{path}: no record file; its worker never ran")
    data = open(path, "rb").read()
    if len(data) < 16:
        raise ko.ObjectError(f"{path}: {len(data)} bytes, no record file header; its worker never opened it")
    magic, version, digest = data[:4], *struct.unpack_from("<IQ", data, 4)
    if magic != ko.RECORD_FILE_MAGIC:
        raise ko.ObjectError(f"{path}: magic {magic!r}, not {ko.RECORD_FILE_MAGIC!r}")
    if version != ko.TOGSIM_VERSION_NUMBER:
        raise ko.ObjectError(f"{path}: version 0x{version:x}, this compiler writes 0x{ko.TOGSIM_VERSION_NUMBER:x}")
    if digest != got["table_hash"]:
        raise ko.ObjectError(f"{path}: table hash 0x{digest:x}, the manifest's is 0x{got['table_hash']:x}: "
                             f"another compile's ELF wrote it")
    out, at = [], 16
    while at < len(data):
        if at + 16 > len(data):
            raise ko.ObjectError(f"{path}: a record header cut at byte {at}")
        key, i, n = struct.unpack_from("<QII", data, at)
        if at + 16 + n > len(data):
            raise ko.ObjectError(f"{path}: record {len(out)} (key {key}, id {i}) of {n} bytes cut at byte {at}")
        out.append((key, i, data[at + 16:at + 16 + n]))
        at += 16 + n + (-n & 7)
    return out


def _values(a, key, body):
    """(extents, values as int64) of one record of attribute `a` for `key`: its extents, then as many
    values of `elem_bits` as they hold -- for a G record, one trip's row."""
    import numpy as np
    rank, bits = a["rank"], a.get("elem_bits", 64)
    if len(body) < 8 * rank:
        raise ko.ObjectError(f"key {key} attribute {a['id']}: {len(body)} bytes cannot hold its {rank} extents")
    extents = [int(e) for e in np.frombuffer(body[:8 * rank], dtype=np.int64)]
    if any(e < 0 for e in extents) or any(e > c for e, c in zip(extents, a["capacity"])):
        raise ko.ObjectError(f"key {key} attribute {a['id']} extents {extents} exceed its static capacity "
                             f"{a['capacity']}")
    count = math.prod(extents[a["fill"]:] if a.get("fill") is not None else extents)
    if len(body) != 8 * rank + count * bits // 8:
        raise ko.ObjectError(f"key {key} attribute {a['id']}: {len(body)} bytes, its extents {extents} of "
                             f"{bits}-bit values need {8 * rank + count * bits // 8}")
    dtype = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}[bits]
    return extents, np.frombuffer(body[8 * rank:], dtype=dtype).astype(np.int64)


def write_runtime_attributes(manifest, runtime, workers):
    """The record files of `workers` under runtime/runtime_attribute/ replaced by what a TOG reads:
    PROGRAMS_FILE (the keys the program markers name) and one attr_<key>_<id>.bin per attribute a
    program recorded, a G attribute's records stacked on its trip dims; ObjectError on any mismatch.
    The keys and the record files' bytes."""
    import numpy as np
    got = manifest["runtime_attributes"]
    by_id = {a["id"]: a for a in got["attributes"]}
    out = os.path.join(runtime, ko.RUNTIME_ATTRIBUTE_DIR)
    keys, grouped, nbytes = set(), {}, 0
    for w in workers:
        path = os.path.join(runtime, ko.record_file(w))
        nbytes += os.path.getsize(path) if os.path.isfile(path) else 0
        program = None
        for key, i, body in read_records(manifest, path):
            if i == ko.RUNTIME_ATTRIBUTE_PROGRAM_ID:
                if body:
                    raise ko.ObjectError(f"{path}: the program marker of key {key} carries {len(body)} bytes")
                if key in keys:
                    raise ko.ObjectError(f"{path}: a second program marker of key {key}")
                keys.add(key)
                program = key
                continue
            if i not in by_id:
                raise ko.ObjectError(f"{path}: key {key} records attribute {i}, which the manifest does not declare")
            if key != program:
                raise ko.ObjectError(f"{path}: key {key} records attribute {i} with no program marker of its "
                                     f"key before it")
            grouped.setdefault((key, i), []).append(body)
    if not keys:
        raise ko.ObjectError("no program ran")
    for w in workers:
        os.remove(os.path.join(runtime, ko.record_file(w)))
    for (key, i), bodies in sorted(grouped.items()):
        a = by_id[i]
        parsed = [_values(a, key, b) for b in bodies]
        extents = parsed[0][0]
        if a.get("fill") is None:
            if len(parsed) != 1:
                raise ko.ObjectError(f"key {key} recorded attribute {i} {len(parsed)} times")
        else:
            if any(e != extents for e, _ in parsed):
                raise ko.ObjectError(f"key {key} attribute {i}: its records' extents differ, "
                                     f"{sorted({tuple(e) for e, _ in parsed})}")
            if len(parsed) != math.prod(extents[:a["fill"]]):
                raise ko.ObjectError(f"key {key} attribute {i} filled {len(parsed)} of "
                                     f"{math.prod(extents[:a['fill']])} trip rows")
        with open(os.path.join(out, f"attr_{key}_{i}.bin"), "wb") as fh:
            np.array([ko.ATTRIBUTE_FILE_MAGIC, ko.TOGSIM_VERSION_NUMBER, i, a["rank"]], dtype=np.uint32).tofile(fh)
            np.array([key], dtype=np.uint64).tofile(fh)
            np.array(extents, dtype=np.int64).tofile(fh)
            np.concatenate([v for _, v in parsed]).astype(np.int64).tofile(fh)
    with open(os.path.join(out, ko.PROGRAMS_FILE), "wb") as fh:
        np.array([ko.PROGRAMS_MAGIC, ko.TOGSIM_VERSION_NUMBER], dtype=np.uint32).tofile(fh)
        np.array([len(keys)] + sorted(keys), dtype=np.uint64).tofile(fh)
    return sorted(keys), nbytes


def _split(manifest, raw_paths, cmd, runtime, total, jobs, log, run, error):
    """The grid as `jobs` processes over ascending disjoint program ranges, each with its own
    --base-path, its own copy of every written .raw and its own record file; the whole grid in one
    process instead when two ranges wrote one byte differently. The workers whose record files hold the run."""
    edges = [(total * w) // jobs for w in range(jobs + 1)]
    ranges = [(edges[w], edges[w + 1]) for w in range(jobs) if edges[w] < edges[w + 1]]
    log(f"[spike] {total} programs over {len(ranges)} processes")

    dirs, cmds = [], []
    start_records(manifest, runtime, range(len(ranges)))
    for w, (lo, hi) in enumerate(ranges):
        d = os.path.join(runtime, f"part{w}")
        _prepare_worker_dir(manifest, raw_paths, d)
        dirs.append(d)
        argv = [_worker_path(manifest, raw_paths, d, runtime, c) for c in cmd]
        argv[argv.index(f"--base-path={runtime}")] = f"--base-path={d}"
        for k, c in enumerate(ko.argv_records(manifest, 0)):
            argv[argv.index(c)] = ko.argv_records(manifest, w)[k]
        cmds.append(argv + [str(lo), str(hi)])

    with cf.ThreadPoolExecutor(max_workers=len(cmds)) as pool:
        for f in cf.as_completed([pool.submit(run, c, runtime) for c in cmds]):
            f.result()

    try:
        merge_worker_outputs(manifest, raw_paths, dirs, error)
    except NotSplittable as why:
        log(f"[spike] {why} -- running the grid in one process instead")
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
        start_records(manifest, runtime, [0])
        run(cmd, runtime)
        return [0]

    merge_access_traces(runtime, dirs, error)
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    return list(range(len(ranges)))


def _worker_path(manifest, raw_paths, worker, runtime, item):
    """A written .raw's basename redirected to the worker's copy, relative to runtime; else `item`."""
    written = [os.path.basename(raw_paths[a["name"]]) for a in ko.writable(manifest)]
    if item in written:
        return os.path.join(os.path.relpath(worker, runtime), item)
    return item


def _prepare_worker_dir(manifest, raw_paths, worker):
    """A worker's own dma_access/ and its own copy of every written .raw."""
    shutil.rmtree(worker, ignore_errors=True)
    os.makedirs(os.path.join(worker, "dma_access"))
    for a in ko.writable(manifest):
        src = raw_paths[a["name"]]
        shutil.copyfile(src, os.path.join(worker, os.path.basename(src)))


def merge_worker_outputs(manifest, raw_paths, workers, error=SpikeError):
    """Fold the workers' copies of each written buffer into the real one: a byte that differs from zero
    was written. Two ranges writing one byte differently raise NotSplittable, since which ran last would
    decide it; a zero stored over another range's value is not seen."""
    import numpy as np

    for a in ko.writable(manifest):
        dest = raw_paths[a["name"]]
        nbytes = ko.nbytes(a)
        base = np.zeros(nbytes, dtype=np.uint8)
        merged, written = base.copy(), np.zeros(nbytes, dtype=bool)
        for d in workers:
            got = np.fromfile(os.path.join(d, os.path.basename(dest)), dtype=np.uint8)
            if got.size != nbytes:
                raise error(f"{d} wrote {got.size} bytes for {a['name']}, expected {nbytes}")
            changed = got != base
            clash = changed & written & (got != merged)
            if clash.any():
                raise NotSplittable(
                    f"two program ranges wrote {a['name']} byte "
                    f"{int(np.flatnonzero(clash)[0])} differently, so which of "
                    f"them ran last decides the answer")
            merged[changed] = got[changed]
            written |= changed
        merged.tofile(dest)


def merge_access_traces(runtime, workers, error=SpikeError):
    """Gather the workers' dma_access/ dumps: fixed names, so more than one range writing them is refused."""
    wrote = [d for d in workers if os.listdir(os.path.join(d, "dma_access"))]
    if len(wrote) > 1:
        raise error("more than one program range wrote dma_access/; it has no per-range "
                    "naming. Run the grid in one process (a job count of 1).")
    for d in wrote:
        for name in os.listdir(os.path.join(d, "dma_access")):
            os.replace(os.path.join(d, "dma_access", name),
                       os.path.join(runtime, "dma_access", name))
