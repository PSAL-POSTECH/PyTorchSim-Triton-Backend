"""Spike, the functional device: the kernel object's ELF over the .raw files of its arguments, the grid
cut into program ranges across processes when no argument is read as well as written. The caller says
where Spike and pk are, how a process is spawned and how a failure is spelled; the rest is kernel.json."""
import concurrent.futures as cf
import os
import re
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


def command(spike, pk, elf, manifest, runtime, raw_paths, options, scalar_values=None):
    """Spike's argv for the whole grid under `options` (kernel_object.run_options): the .raw paths as
    basenames, since pk's argv is a fixed 2048 bytes and Spike runs with cwd=runtime; then the scalars."""
    lo, hi = manifest["abi"]["kernel_addr"]
    return [spike, "--isa", options["spike_isa"], f"--extlib={options['model']}",
            "-m" + options["memory"], f"--machine-config={options['machine_config']}",
            f"--kernel-addr={lo:x}:{hi:x}", f"--base-path={runtime}", pk, elf,
            *(os.path.basename(raw_paths[n]) for n in ko.argv_order(manifest)),
            *ko.argv_tail(manifest, scalar_values)]


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
    """Run the whole grid, leaving each written argument in its .raw; `run(cmd, cwd)` spawns one
    process (run_process by default) and `error(message)` is the exception a failed merge raises."""
    run = run or (lambda cmd, cwd: run_process(cmd, cwd, log))
    shutil.rmtree(os.path.join(runtime, "indirect_access"), ignore_errors=True)
    os.makedirs(os.path.join(runtime, "indirect_access"))
    os.makedirs(os.path.join(runtime, "dma_access"), exist_ok=True)
    cmd = command(spike, pk, elf, manifest, runtime, raw_paths, options, scalar_values)
    total = programs(manifest)
    n = jobs(manifest, total, log, want_jobs)
    if n > 1:
        _split(manifest, raw_paths, cmd, runtime, total, n, log, run, error)
    else:
        run(cmd, runtime)
    return runtime


def _split(manifest, raw_paths, cmd, runtime, total, jobs, log, run, error):
    """The grid as `jobs` processes over ascending disjoint program ranges, each with its own
    --base-path and its own copy of every written .raw; the whole grid in one process instead when
    two ranges wrote one byte differently."""
    edges = [(total * w) // jobs for w in range(jobs + 1)]
    ranges = [(edges[w], edges[w + 1]) for w in range(jobs) if edges[w] < edges[w + 1]]
    log(f"[spike] {total} programs over {len(ranges)} processes")

    workers, cmds = [], []
    for w, (lo, hi) in enumerate(ranges):
        d = os.path.join(runtime, f"part{w}")
        _prepare_worker_dir(manifest, raw_paths, d)
        workers.append(d)
        argv = [_worker_path(manifest, raw_paths, d, runtime, c) for c in cmd]
        argv[argv.index(f"--base-path={runtime}")] = f"--base-path={d}"
        cmds.append(argv + [str(lo), str(hi)])

    with cf.ThreadPoolExecutor(max_workers=len(cmds)) as pool:
        for f in cf.as_completed([pool.submit(run, c, runtime) for c in cmds]):
            f.result()

    try:
        merge_worker_outputs(manifest, raw_paths, workers, error)
    except NotSplittable as why:
        log(f"[spike] {why} -- running the grid in one process instead")
        for d in workers:
            shutil.rmtree(d, ignore_errors=True)
        run(cmd, runtime)
        return

    merge_access_traces(runtime, workers, error)
    for d in workers:
        shutil.rmtree(d, ignore_errors=True)


def _worker_path(manifest, raw_paths, worker, runtime, item):
    """A written .raw's basename redirected to the worker's copy, relative to runtime; else `item`."""
    for a in ko.writable(manifest):
        if item == os.path.basename(raw_paths[a["name"]]):
            return os.path.join(os.path.relpath(worker, runtime), item)
    return item


def _prepare_worker_dir(manifest, raw_paths, worker):
    """A worker's own dump directories and its own copy of every written .raw."""
    shutil.rmtree(worker, ignore_errors=True)
    os.makedirs(os.path.join(worker, "indirect_access"))
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
    """Gather the workers' index dumps: a keyed one (indirect_index_<key>_<n>.raw) by name, an unkeyed
    one renumbered worker by worker; dma_access/ has fixed names, so only one range may write it."""
    dst = os.path.join(runtime, "indirect_access")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)

    n = 0
    for d in workers:
        src = os.path.join(d, "indirect_access")
        names = os.listdir(src)
        unkeyed = sorted((f for f in names if re.fullmatch(r"indirect_index\d+\.raw", f)),
                         key=lambda f: int(re.findall(r"\d+", f)[-1]))
        for name in names:
            if name not in unkeyed:
                os.replace(os.path.join(src, name), os.path.join(dst, name))
        for name in unkeyed:
            os.replace(os.path.join(src, name), os.path.join(dst, f"indirect_index{n}.raw"))
            n += 1

    wrote = [d for d in workers if os.listdir(os.path.join(d, "dma_access"))]
    if len(wrote) > 1:
        raise error("more than one program range wrote dma_access/; it has no per-range "
                    "naming. Run the grid in one process (a job count of 1).")
    for d in wrote:
        for name in os.listdir(os.path.join(d, "dma_access")):
            os.replace(os.path.join(d, "dma_access", name),
                       os.path.join(runtime, "dma_access", name))
