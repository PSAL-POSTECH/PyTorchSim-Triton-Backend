"""Spike -- the functional device: run the ELF over the tensors on disk.

Reads the kernel object's manifest for the kernel's address range, the launch
grid and each argument's role, and the target for the machine.
"""
import concurrent.futures as cf
import os
import re
import shutil

from pytorchsim_triton_compiler.contract import kernel_object as ko
from pytorchsim_triton_compiler.contract import target as _target
from . import config
from .base import Device


class SpikeDevice(Device):
    """Runs the whole launch grid, splitting it across processes when it can."""

    name = "spike"

    def launch(self, req):
        return _launch(req)


def _launch(req):
    elf = req.artifact("elf")
    runtime = req.runtime_dir
    shutil.rmtree(os.path.join(runtime, "indirect_access"), ignore_errors=True)
    os.makedirs(os.path.join(runtime, "indirect_access"))
    os.makedirs(os.path.join(runtime, "dma_access"), exist_ok=True)

    # BUILDING THE TENSORS IS NOT RUNNING THE KERNEL, and on a small kernel it
    # is the larger half: this is where torch is first imported and where the
    # reference inputs are generated and written. Charging it to "spike" reads
    # as the simulator being slow when the simulator has not started.
        # BASENAMES, NOT ABSOLUTE PATHS, and spike is run with cwd=runtime below.
    # These land in pk's argv, whose buffer is a fixed 2048 bytes
    # (riscv-pk/pk/pk.c: `panic("args must not exceed %d bytes")`). The
    # directory prefix is the same for every one of them, so carrying it here
    # spends the budget on a constant: measured on RecurrentGemma's Griffin
    # block, whose 24 arguments came to 2871 bytes of argv against 2048, while
    # every other kernel in that model sat between 734 and 1111 with 2 to 5
    # arguments. The names themselves are ~10 bytes each.
    argv_paths = [os.path.basename(req.raw_paths[n])
                  for n in ko.argv_order(req.manifest)] + ko.argv_tail(req.manifest)

    t = _target.parse(req.manifest["target"], f"{req.obj_dir}: manifest target")
    problems = t.memory_map_problems()
    if problems:
        raise req.error("scratchpad memory map is unusable:\n  - "
                        + "\n  - ".join(problems))

    spad_bytes = t.banked_bytes()
    lo, hi = req.manifest["abi"]["kernel_addr"]
    yml = config.MACHINE_CONFIG or t.write_model_yaml(
        os.path.join(runtime, "machine.yml"))
    machine = [f"--machine-config={yml}"]
    cmd = [
        config.SPIKE,
        "--isa", config.SPIKE_ISA,
        f"--extlib={config.vcix_model(t)}",
        "-m" + f"0x{t.dram_base:x}:0x{t.dram_size:x},"
               f"0x{t.spad_paddr:x}:0x{spad_bytes:x}",
        *machine,
        f"--kernel-addr={lo:x}:{hi:x}",
        f"--base-path={runtime}",
        config.PK, elf, *argv_paths,
    ]

    total = 1
    for g in ko.grid3(req.manifest["grid"]):
        total *= g
    jobs = _spike_jobs(req, total)

    try:
        with req.step("run the grid"):
            if jobs > 1:
                _spike_split(req, cmd, runtime, total, jobs)
            else:
                req.run("spike", cmd, cwd=runtime)
    except Exception as e:
        hint = {200: "INVALID_SPAD_ACCESS", 201: "STACK_OVERFLOW"}
        for code, name in hint.items():
            if f"exit code {code}" in str(e):
                e.args = (f"{e.args[0]}  ({name})",)
        raise
    req.log("[spike] spike finished")
    return runtime


def _spike_jobs(req, total):
    """How many processes to spread the launch grid over.

    Spike interprets one instruction at a time, so a kernel's stage-6 cost is
    the whole grid walked serially -- ResNet's first conv is 196 programs at
    about a second each. Triton's programs are independent by contract, so the
    range can be cut up.

    EXCEPT WHERE A VALUE CROSSES BETWEEN THEM. Two programs writing one address
    is fine and needs no special handling: the workers hold ascending contiguous
    ranges and the merge applies them in that order, so the last write wins
    exactly as it does in one sequential walk. A program READING what another
    wrote is not fine, and no merge can repair it -- the worker never saw the
    write, so it read the initial value and computed something else.

    That hazard needs a buffer that is both written and read, which is what
    `inout` says, so an `inout` argument is where this stops. It is the only
    signal the spec carries. THE GAP IT LEAVES: a kernel that loads from an
    `out` buffer another program stored to is still wrong here, and nothing in
    the role says so. Nothing in the suite does it (293 kernels, identical
    results split and whole); a kernel that starts to would need this to become
    a question about the IR rather than about the signature.

    `TORCHSIM_COMPILE_SPIKE_JOBS` overrides the count, and 1 restores the single process.
    The default stays well under the core count because a caller (PyTorchSim's
    sweep, the regression runner) usually has several kernels in flight already.
    """
    inout = [a["name"] for a in req.manifest["args"] if a["role"] == "inout"]
    if not ko.splittable(req.manifest):
        req.log(f"[spike] one process: {', '.join(inout)} is read as well as "
                f"written, so a program may depend on another one's store")
        return 1
    env = os.environ.get("TORCHSIM_COMPILE_SPIKE_JOBS", "").strip()
    if env:
        return max(1, min(int(env), total))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:                       # not linux
        cpus = os.cpu_count() or 1
    return max(1, min(8, cpus // 2, total))


def _spike_split(req, cmd, runtime, total, jobs):
    """Run the grid as `jobs` processes over disjoint program ranges.

    Each worker gets its own --base-path, because spike dumps its indirect-DMA
    index vectors there and every process numbers them from zero. Inputs are
    shared read-only; anything the kernel writes is a per-worker copy, since the
    wrapper dumps the WHOLE buffer and a worker only filled its own slice of it.
    """
    edges = [(total * w) // jobs for w in range(jobs + 1)]
    ranges = [(edges[w], edges[w + 1]) for w in range(jobs)
              if edges[w] < edges[w + 1]]
    req.log(f"[spike] {total} programs over {len(ranges)} processes")

    workers, cmds = [], []
    for w, (lo, hi) in enumerate(ranges):
        d = os.path.join(runtime, f"part{w}")
        _prepare_worker_dir(req, runtime, d)
        workers.append(d)
        # cmd is [...spike opts..., --base-path=runtime, pk, elf, *raw paths].
        # The base path and the writable raw paths move to the worker; the
        # bounds go on the end, where the wrapper reads them.
        argv = [_worker_path(req, d, runtime, c) for c in cmd]
        argv[argv.index(f"--base-path={runtime}")] = f"--base-path={d}"
        cmds.append(argv + [str(lo), str(hi)])

    with cf.ThreadPoolExecutor(max_workers=len(cmds)) as pool:
        futures = [pool.submit(req.run, "spike", c, cwd=runtime)
                   for c in cmds]
        for f in cf.as_completed(futures):
            f.result()                            # the first failure wins

    try:
        _merge_worker_outputs(req, workers)
    except _NotSplittable as why:
        # The workers wrote only their own copies, so the real buffers still
        # hold what the caller put there and one process starting now produces
        # exactly what it always did. Refusing instead would turn a stage-6
        # speedup into a per-kernel setting for every kernel that overwrites an
        # output, and those are the kernels most worth running.
        req.log(f"[spike] {why} -- running the grid in one process instead")
        for d in workers:
            shutil.rmtree(d, ignore_errors=True)
        req.run("spike", cmd, cwd=runtime)
        return

    _merge_access_traces(req, runtime, workers)
    for d in workers:
        shutil.rmtree(d, ignore_errors=True)


class _NotSplittable(Exception):
    """Which program ran last decides the answer, so the split cannot give it."""


def _writable_args(req):
    return ko.writable(req.manifest)


def _worker_path(req, worker, runtime, item):
    """Redirect a written .raw to the worker's copy; leave everything else.

    Both sides are basenames now (see argv_paths), so the match is on the name
    and the redirect stays RELATIVE to runtime -- `part0/out.raw` -- because
    that is the cwd every worker is run with.
    """
    for a in _writable_args(req):
        if item == os.path.basename(req.raw_paths[a["name"]]):
            return os.path.join(os.path.relpath(worker, runtime), item)
    return item


def _prepare_worker_dir(req, runtime, worker):
    shutil.rmtree(worker, ignore_errors=True)
    os.makedirs(os.path.join(worker, "indirect_access"))
    os.makedirs(os.path.join(worker, "dma_access"))
    for a in _writable_args(req):
        src = req.raw_paths[a["name"]]
        shutil.copyfile(src, os.path.join(worker, os.path.basename(src)))


def _merge_worker_outputs(req, workers):
    """Fold each worker's copy of a written buffer back into the real one.

    A worker's buffer starts where the sequential run's would have -- zeroed,
    which is what main() callocs and never loads for an `out` argument -- so a
    byte it changed is a byte its programs wrote.

    Applying the workers in order would seem to be enough -- they hold ascending
    contiguous ranges, so the last write to an address wins either way. IT IS
    NOT, and the reason is that "wrote" here is inferred rather than recorded: a
    byte counts as written when it differs from the start, so a program storing
    a ZERO over an earlier program's value looks exactly like a program that
    left it alone, and the earlier value survives a merge that one sequential
    walk would have overwritten. MEASURED: merging in order alone moves
    mask_gs_from_structured off bit-identical (0 -> 1.00117e-08) and breaks
    reject/ptr_store_wrap outright (mismatch 5.41361).

    So an address two ranges disagree about is refused here and the caller runs
    the grid whole, where the answer is defined. Both of those kernels store
    through a wrapped pointer, both are caught, and the suite comes back
    byte-identical to the sequential run.

    WHAT THIS DOES NOT SEE, for the same reason as above: two ranges whose
    overlap is invisible because the later one stored zeros. Nothing in the
    suite does that. Recording the writes instead of inferring them is what
    would settle it, and that is the wrapper's to do, not this function's.
    """
    import numpy as np

    for a in _writable_args(req):
        dest = req.raw_paths[a["name"]]
        nbytes = ko.nbytes(a)
        base = np.zeros(nbytes, dtype=np.uint8)
        merged, written = base.copy(), np.zeros(nbytes, dtype=bool)
        for d in workers:
            got = np.fromfile(os.path.join(d, os.path.basename(dest)),
                              dtype=np.uint8)
            if got.size != nbytes:
                raise req.error(f"{d} wrote {got.size} bytes for "
                                f"{a['name']}, expected {nbytes}")
            changed = got != base
            clash = changed & written & (got != merged)
            if clash.any():
                raise _NotSplittable(
                    f"two program ranges wrote {a['name']} byte "
                    f"{int(np.flatnonzero(clash)[0])} differently, so which of "
                    f"them ran last decides the answer")
            merged[changed] = got[changed]
            written |= changed
        merged.tofile(dest)


def _merge_access_traces(req, runtime, workers):
    """Gather the workers' index dumps: a keyed one (indirect_index_<key>_<n>.raw, the
    wrapper's dma_index_key) by name, since the key names its program; an unkeyed one
    renumbered worker by worker, the order one process would have written them in."""
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
            os.replace(os.path.join(src, name),
                       os.path.join(dst, f"indirect_index{n}.raw"))
            n += 1

    # Fixed names, so there is nothing to renumber and no way to tell two
    # workers' apart. Nothing on this route writes them; refuse if that changes.
    wrote = [d for d in workers if os.listdir(os.path.join(d, "dma_access"))]
    if len(wrote) > 1:
        raise req.error("more than one program range wrote "
                        "dma_access/; it has no per-range naming. "
                        "Set TORCHSIM_COMPILE_SPIKE_JOBS=1.")
    for d in wrote:
        for name in os.listdir(os.path.join(d, "dma_access")):
            os.replace(os.path.join(d, "dma_access", name),
                       os.path.join(runtime, "dma_access", name))
