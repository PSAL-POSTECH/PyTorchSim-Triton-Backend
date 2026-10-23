"""The ELF Spike runs: LLVM IR to object, the C driver beside it, the link, and the budgets the
result is held to."""
import collections
import os
import re
import shutil
import subprocess

import yaml

from ... import machine, toolchain
from ...contract import kernel_object
from ...metrics import quality
from ...stage import SpadOverflow, SpillNotAllowed, StageError, run_tool
from . import wrapper


_LL_ELEM_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "i128": 128,
                 "half": 16, "bfloat": 16, "float": 32, "double": 64,
                 "ptr": 64}


_LL_VECTOR = re.compile(r"<(\d+) x ([a-z0-9]+)>")


_ELF_SECTION = re.compile(
    r"^\s*\[\s*\d+\]\s+(?P<name>\S+)\s+\S+\s+[0-9a-f]+\s+"
    r"[0-9a-f]+\s+(?P<size>[0-9a-f]+)", re.MULTILINE)


def _measure_spad(elf, readelf):
    from ..passes import p13_lower_to_llvm as p13
    r = subprocess.run([readelf, "-SW", elf], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"readelf failed on {elf}: {r.stderr.strip()}")
    for m in _ELF_SECTION.finditer(r.stdout):
        if m.group("name") == p13.SPAD_SECTION:
            return int(m.group("size"), 16)
    return 0


def _check_spad_budget(ctx, elf, stage, tag):
    spad_size, spad_budget = machine.SPAD_SIZE, machine.target().spad_budget
    try:
        usage = _measure_spad(elf, toolchain.RISCV_READELF)
    except RuntimeError as e:
        ctx.warn(f"could not measure .spad usage: {e}")
        return 0
    said = None
    if usage > spad_size:
        budget, said = spad_size, (
            f"scratchpad overflow: kernel claims {usage} bytes/lane but Spike maps "
            f"only {spad_size} (--scratchpad-size). Raise TORCHSIM_COMPILE_SPAD_SIZE or shrink "
            f"the tile.")
    elif usage > spad_budget:
        budget, said = spad_budget, (
            f"scratchpad usage {usage} bytes/lane exceeds the double-buffer "
            f"budget of {spad_budget} (spad / vpu_spad_buffers), so this kernel cannot be "
            f"double buffered on the target. THERE ARE TWO SHAPES OF THIS and "
            f"the buffer count tells them apart: a handful of globals means ONE "
            f"of them is wrong -- dump the memref.globals and look for a tile "
            f"banked on a unit axis, which lands the whole thing in one lane and "
            f"reserves it in every lane -- while dozens of globals at a few KB "
            f"each means the TILE is too big for what the kernel fuses, and the "
            f"caller that chose the block size is the one that can fix it "
            f"(PyTorchSim's Inductor route reads usage and budget from failure.json "
            f"and retries with a smaller R0_BLOCK). Raise TORCHSIM_COMPILE_SPAD_SIZE only if "
            f"the tile is genuinely that big.")
    if said is not None:
        raise SpadOverflow(stage, said, usage=usage, budget=budget, artifact=elf)
    if usage:
        ctx.log(f"[{tag}] scratchpad: {usage} bytes/lane of {spad_size}")
    return usage


def _check_vector_budget(ctx, ll, stage="binary"):
    budget = machine.VLEN_BITS * machine.ISA_MAX_LMUL
    with open(ll) as f:
        text = f.read()
    text = re.sub(r"(shufflevector\b[^\n]*?), <\d+ x i32> [^\n]*", r"\1", text)
    worst = {}
    for n, elem in _LL_VECTOR.findall(text):
        bits = _LL_ELEM_BITS.get(elem)
        if bits is None:
            continue
        total = int(n) * bits
        if total > budget:
            worst[f"<{n} x {elem}>"] = total
    if worst:
        shown = "; ".join(
            f"{v} needs {b} bits (LMUL {b // machine.VLEN_BITS})"
            for v, b in sorted(worst.items(), key=lambda kv: -kv[1])[:3])
        raise StageError(
            stage,
            f"a vector is wider than one instruction can reach: {shown}. The "
            f"budget is VLEN {machine.VLEN_BITS} x LMUL {machine.ISA_MAX_LMUL} = "
            f"{budget} bits. The element count is chosen against the type a "
            f"chain STARTS with (legalize) and spent "
            f"against the widest type it reaches, so a widening op after the "
            f"choice is what usually does this.",
            artifact=ll)


_CFA = re.compile(r"\.cfi_def_cfa_offset\s+(\d+)|\.cfi_escape[^#\n]*#\s*sp\s*\+"
                  r"\s*(\d+)(?:\s*\+\s*(\d+)\s*\*\s*vlenb)?")


def _get_stack_depth(text):
    depth = 0
    for m in _CFA.finditer(text):
        got = (int(m.group(1)) if m.group(1) else int(m.group(2))
               + int(m.group(3) or 0) * machine.VLEN_BITS // 8)
        depth = max(depth, got)
    return depth


class _Remark(yaml.SafeLoader):
    """llc's -pass-remarks-output: one tagged mapping (!Missed, !Passed, !Analysis) per remark."""


for _tag in ("!Missed", "!Passed", "!Analysis", "!Failure", "!AnalysisFPCommute", "!AnalysisAliasing"):
    _Remark.add_constructor(_tag, lambda loader, node: loader.construct_mapping(node, deep=True))


def _spills_and_reloads(remarks):
    """The spills and reloads the register allocator reports for the whole module: the sum of every
    function's SpillReloadCopies remark (the per-loop remarks repeat parts of it)."""
    total = 0
    with open(remarks) as f:
        for r in yaml.load_all(f, Loader=_Remark):
            if r and r.get("Pass") == "regalloc" and r.get("Name") == "SpillReloadCopies":
                for arg in r.get("Args", []):
                    total += int(arg.get("NumSpills", 0)) + int(arg.get("NumReloads", 0))
    return total


VectorSpills = collections.namedtuple("VectorSpills", "spills reloads loops dots outside units")


def _vector_args(remark):
    return {k: int(v) for a in remark.get("Args", []) for k, v in a.items()
            if k in ("NumVectorSpills", "NumVectorReloads", "VectorSpilledRegisterUnits")}


def read_places(remarks):
    """The vector registers' spills and reloads llc's regalloc remarks count (NumVectorSpills,
    NumVectorReloads; fork L8), as VectorSpills: the module's spills and reloads, from each function's
    SpillReloadCopies; by location, from its per-line BlockSpillReloadCopies, the loops' own ops and the
    matmul streams carrying p06_legalize.LOOP_FILE and lib_matmul.DOT_FILE ({loop number: its spills and
    reloads}, {matmul number: ...}, {("loop" | "dot", number): the vector register units it spilled},
    VectorSpilledRegisterUnits); and how many neither names. Each allocation run reports its whole
    function and ends in that function's SpillReloadCopies, so a function's last run is the one read."""
    from ..passes.lib_matmul import DOT_FILE
    from ..passes.p06_legalize import LOOP_FILE
    lines, last = {}, {}
    with open(remarks) as f:
        for r in yaml.load_all(f, Loader=_Remark):
            if not r or r.get("Pass") != "regalloc":
                continue
            fn = r.get("Function")
            if r.get("Name") == "BlockSpillReloadCopies":
                lines.setdefault(fn, []).append(r)
            elif r.get("Name") == "SpillReloadCopies":
                last[fn] = (_vector_args(r), lines.pop(fn, []))
    spills = reloads = named = 0
    loops, dots, units = {}, {}, {}
    for whole, run in last.values():
        spills += whole.get("NumVectorSpills", 0)
        reloads += whole.get("NumVectorReloads", 0)
        for r in run:
            args, at = _vector_args(r), r.get("DebugLoc") or {}
            spilled = args.get("NumVectorSpills", 0) + args.get("NumVectorReloads", 0)
            kind = {LOOP_FILE: "loop", DOT_FILE: "dot"}.get(at.get("File"))
            if not spilled or kind is None:
                continue
            n = int(at["Line"])
            into = loops if kind == "loop" else dots
            into[n] = into.get(n, 0) + spilled
            units[(kind, n)] = units.get((kind, n), 0) + args.get("VectorSpilledRegisterUnits", 0)
            named += spilled
    return VectorSpills(spills, reloads, loops, dots, max(0, spills + reloads - named), units)


def narrowable(ctx, loops, dots):
    """The places llc names for a spill that a narrower width may relieve: a loop or a matmul stream
    wider than one register the retry has not found its width does nothing for (ctx.spill_kept)."""
    out = {("loop", n): k for n, k in loops.items() if ctx.loop_lmul.get(n, ctx.lmul) > 1}
    out.update({("dot", n): k for n, k in dots.items() if ctx.dot_widths.get(n, 1) > 1})
    return {p: k for p, k in out.items() if p not in ctx.spill_kept}


def _check_stack_is_allowed(ctx, asm, obj, usage):
    with open(asm) as f:
        text = f.read()
    spilled = read_places(asm + ".regalloc.yaml")
    stores, loads = spilled.spills, spilled.reloads
    depth = _get_stack_depth(text)
    fits = usage + depth <= machine.SPAD_SIZE
    if stores or loads:
        ctx.log(f"[binary] llc spilled {stores} vector register group(s) "
                f"and reloads {loads} at LMUL {ctx.lmul}; the stack is "
                f"{depth} bytes over {usage} bytes/lane of tiles in a "
                f"{machine.SPAD_SIZE}-byte bank")
        # A spill a narrower loop or matmul stream may relieve is not allowed even where the stack
        # fits: the retry narrows the place llc names for the most of them, one at a time, and keeps
        # a width that relieves nothing (pipeline._narrow_for). What no narrowing relieves is allowed
        # where the stack fits, as any spill is.
        loops, dots, outside, units = spilled.loops, spilled.dots, spilled.outside, spilled.units
        relieve = narrowable(ctx, loops, dots)
        if not relieve and (loops or dots or outside):
            ctx.log(f"[binary] spills no narrower loop or stream relieves: loops {loops or 'none'}, "
                    f"streams {dots or 'none'}, where neither is named {outside}")
        if not fits or relieve or ctx.spill_last is not None:
            ctx.log("[binary] spill not allowed: " + (
                "the stack does not fit above the tiles" if not fits
                else "a narrower loop or matmul stream may relieve them" if relieve
                else "the last narrowing is judged"))
            err = SpillNotAllowed(
                "binary",
                f"llc spilled {stores} vector register group(s) "
                + ("even at LMUL 1" if ctx.lmul == 1 else f"at LMUL {ctx.lmul}"),
                artifact=obj)
            err.loops, err.dots, err.outside = sorted(loops), sorted(dots), outside
            err.loop_spills, err.dot_spills, err.spilled_units = loops, dots, units
            ctx.log(f"[binary] loops that spill or reload: {loops or 'none located'}; matmul "
                    f"streams: {dots or 'none located'}; where neither is named: {outside}")
            raise err
    if not fits:
        raise StageError(
            "binary",
            f"the kernel's stack, {depth} bytes, does not fit above its tiles, "
            f"{usage} bytes/lane, in a {machine.SPAD_SIZE}-byte bank: spike "
            f"lays the stack on the top of each lane's bank and stops with "
            f"VI_ST_WHOLE ERROR: Accessing invalid spad address or Accessing "
            f"stack address.", artifact=asm)


def build_elf(ctx, stage, tag, lowered, ll, obj, asm, wrap_c, elf,
              single_program=False):
    """LLVM-dialect MLIR -> `(ELF, .spad bytes/lane)`, with the vector and scratchpad budgets
    checked."""
    run_tool(ctx, stage, [toolchain.MLIR_TRANSLATE, "-mlir-to-llvmir", lowered, "-o", ll])
    # A loop as scf.for leaves it is tested at its head, so its preheader branches around it and
    # llc's vsetvli insertion, finding the vector state unknown there (a TOG marker or a DMA is
    # inline asm), sets it again in every iteration; rotated, the loop gets it once before.
    # Rotation lifts values into the preheader and can cost registers: when the register allocator
    # reports more spills and reloads for the rotated loops than for the loops as they were, the
    # loops as they were are kept, at this LMUL, rather than the kernel going back to a narrower one.
    plain = ll + ".unrotated"
    shutil.copy(ll, plain)
    run_tool(ctx, stage, [toolchain.OPT, "-S", "-passes=loop-rotate", plain, "-o", ll])
    _check_vector_budget(ctx, ll, stage)
    t = machine.target()
    toolchain.check_llc_honours(t)
    common = ["-relocation-model=pic", f"-march={t.llc_march()}", "-O3", "--stack-size-section",
              f"-mattr={t.llc_mattr()}", f"--riscv-v-vector-bits-max={t.vlen_bits}", ll]
    remarks = asm + ".regalloc.yaml"
    regalloc = [f"-pass-remarks-output={remarks}", "-pass-remarks-filter=regalloc"]
    run_tool(ctx, stage, [toolchain.LLC, *common, *regalloc, "-o", asm])
    rotated = _spills_and_reloads(remarks)
    if rotated:
        kept = ll + ".rotated"
        shutil.copy(ll, kept)
        shutil.copy(plain, ll)
        run_tool(ctx, stage, [toolchain.LLC, *common, *regalloc, "-o", asm])
        unrotated = _spills_and_reloads(remarks)
        if unrotated < rotated:
            ctx.log(f"[binary] the rotated loops spill and reload {rotated} times, the loops as they "
                    f"were {unrotated}; the loops as they were are kept")
        else:
            shutil.copy(kept, ll)
            run_tool(ctx, stage, [toolchain.LLC, *common, *regalloc, "-o", asm])
    run_tool(ctx, stage, [toolchain.LLC, *common, "-filetype=obj", "-o", obj])

    _write_wrapper(ctx, stage, tag, lowered, wrap_c, single_program)

    from ..passes import p13_lower_to_llvm as p13
    link = [toolchain.RISCV_GCC, "-O3", f"-march={t.gcc_march()}", f"-mabi={t.gcc_mabi()}",
            wrap_c, obj, "-o", elf,
            f"-Wl,--section-start={p13.SPAD_SECTION}"
            f"=0x{t.spad_vaddr:x}",
            f"-Wl,--section-start={p13.SPAD_TOP_SECTION}=0x{t.spad_vaddr + t.banked_bytes():x}"]
    run_tool(ctx, stage, link)

    usage = _check_spad_budget(ctx, elf, stage, tag)
    return elf, usage


def _write_wrapper(ctx, stage, tag, lowered, wrap_c, single_program):
    """The C driver at `wrap_c`: the kernel's hand-written one, or one generated, refilling the
    poison when p13.poisoned(`lowered`) says the module carries it -- refused for a hand-written one."""
    from ..passes import p13_lower_to_llvm as p13
    toolchain.ensure_bindings()
    poisoned = p13.poisoned(lowered)
    hidden = None if single_program else runtime_attribute_table(lowered)
    override = wrapper.find_override(ctx.kernel_path)
    if override and single_program:
        raise StageError(
            stage,
            f"{os.path.basename(override)} is a hand-written wrapper, and it "
            f"runs the whole launch grid. The cycle binary times ONE tile, so "
            f"using it would multiply every count by the grid, and generating "
            f"one instead would ignore whatever the override exists to say. "
            f"Write a single-program wrapper for this kernel, or drop the "
            f"override.", artifact=override)
    if override and hidden:
        raise StageError(
            stage,
            f"{os.path.basename(override)} is a hand-written wrapper, and the kernel records runtime "
            f"attributes into a hidden output it does not write.", artifact=override)
    if override and poisoned:
        raise StageError(
            stage,
            f"{os.path.basename(lowered)} was lowered with {p13.SPAD_POISON_ENV} on, and "
            f"{os.path.basename(override)} is a hand-written wrapper that does not refill the "
            f"poison in every lane before every program: lane 0 would start poisoned and the "
            f"rest zero. Lower again with {p13.SPAD_POISON_ENV} unset, or drop the override.",
            artifact=override)
    if override:
        shutil.copyfile(override, wrap_c)
        ctx.log(f"[{tag}] using hand-written wrapper {override}")
        return
    with open(wrap_c, "w") as f:
        f.write(wrapper.generate(ctx.spec, machine.target(), single_program=single_program,
                                 poisoned=poisoned, runtime_attributes=hidden))
    ctx.log(f"[{tag}] generated wrapper (see {os.path.basename(wrap_c)} "
            f"for the assumed ABI)" + (", refilling the .spad poison" if poisoned else ""))


def runtime_attribute_table(lowered):
    """The slot table the lowered module at `lowered` states (p11b's TABLE), or None."""
    import json
    from mlir.ir import Context, Module, StringAttr
    from ..passes.p11b_lower_runtime_attributes import TABLE
    toolchain.ensure_bindings()
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(lowered) as f:
        attrs = Module.parse(f.read()).operation.attributes
        return json.loads(StringAttr(attrs[TABLE]).value) if TABLE in attrs else None


def _relative_if_present(ctx, name):
    return name if os.path.exists(ctx.path(name)) else None


def _kernel_addr_range(ctx, elf):
    out = run_tool(ctx, "binary", [toolchain.RISCV_NM, "-S", elf]).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[3] == ctx.spec.name:
            start, size = int(parts[0], 16), int(parts[1], 16)
            ctx.log(f"[binary] kernel {ctx.spec.name} at "
                    f"0x{start:x}..0x{start + size:x}")
            return [start, start + size]
    raise StageError(
        "binary",
        f"no sized symbol '{ctx.spec.name}' in the ELF -- cannot tell a device "
        f"the kernel PC range, and without it the vector unit runs single-lane",
        artifact=elf)


def build_kernel_object(ctx):
    """The ELF a device runs, linked and checked, and the manifest naming it."""
    ll = ctx.path("11-kernel.ll")
    asm = ctx.path("11-kernel.s")
    wrap_c = ctx.path("11-wrapper.c")
    obj = ctx.path("11-kernel.o")
    elf, usage = build_elf(ctx, "binary", "stage6",
                           lowered=ctx.path("10-lowered.mlir"),
                           ll=ll, obj=obj, asm=asm, wrap_c=wrap_c,
                           elf=ctx.path(f"11-{ctx.spec.name}.elf"))
    _check_stack_is_allowed(ctx, asm, obj, usage)

    manifest = kernel_object.write(
        ctx.workdir, ctx.spec, machine.target(),
        kernel_addr=_kernel_addr_range(ctx, elf),
        runtime_attributes=runtime_attribute_table(ctx.path("10-lowered.mlir")),
        artifacts={"elf": os.path.basename(elf),
                    "post_vcix_ir": _relative_if_present(ctx, "09-custom.mlir"),
                    "llvm_ir": os.path.basename(ll),
                    "asm": os.path.basename(asm),
                    "wrapper_c": os.path.basename(wrap_c),
                    "quality": _relative_if_present(ctx, quality.FILENAME)})

    ctx.log(f"[binary] ELF: {elf}")
    ctx.log(f"[binary] kernel object: {manifest}")
    return elf
