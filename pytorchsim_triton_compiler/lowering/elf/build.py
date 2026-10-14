"""The ELF Spike runs: LLVM IR to object, the C driver beside it, the link, and the budgets the
result is held to."""
import os
import re
import shutil
import subprocess

from ... import machine, toolchain
from ...contract import kernel_object
from ...metrics import quality
from ...stage import SpillNotAllowed, StageError, run_tool
from . import wrapper


_LANE_EXTRACT = ("vfmv.f.s", "vmv.x.s")


def _check_no_lane_extract(ctx, asm, stage="binary"):
    import re
    with open(asm) as f:
        bad = [(n, line.strip()) for n, line in enumerate(f, 1)
               if re.match(r"\s*(%s)\b" % "|".join(
                   i.replace(".", r"\.") for i in _LANE_EXTRACT), line)]
    if bad:
        where = "; ".join(f"{asm}:{n}: {t}" for n, t in bad[:4])
        raise StageError(
            stage,
            f"vector->scalar extract in a lane-banked kernel: {where}. There is "
            f"one scalar register file for all lanes, so this keeps lane 0 and "
            f"discards the rest. Keep the value in a vector register.",
            artifact=asm)


_LL_ELEM_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "i128": 128,
                 "half": 16, "bfloat": 16, "float": 32, "double": 64,
                 "ptr": 64}


_LL_VECTOR = re.compile(r"<(\d+) x ([a-z0-9]+)>")


SPAD_OVERFLOW_MARKER = "torchsim-spad-overflow"


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
    spad_size = machine.SPAD_SIZE
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
    elif usage > spad_size // 2:
        budget, said = spad_size // 2, (
            f"scratchpad usage {usage} bytes/lane exceeds the double-buffer "
            f"budget of {spad_size // 2} (spad/2), so this kernel cannot be "
            f"double buffered on the target. THERE ARE TWO SHAPES OF THIS and "
            f"the buffer count tells them apart: a handful of globals means ONE "
            f"of them is wrong -- dump the memref.globals and look for a tile "
            f"banked on a unit axis, which lands the whole thing in one lane and "
            f"reserves it in every lane -- while dozens of globals at a few KB "
            f"each means the TILE is too big for what the kernel fuses, and the "
            f"caller that chose the block size is the one that can fix it "
            f"(PyTorchSim's Inductor route reads the marker below and retries "
            f"with a smaller R0_BLOCK). Raise TORCHSIM_COMPILE_SPAD_SIZE only if the tile is "
            f"genuinely that big.")
    if said is not None:
        raise StageError(stage, f"{said}\n{SPAD_OVERFLOW_MARKER}: "
                                f"usage={usage} budget={budget}", artifact=elf)
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


_VEC_SPILL = re.compile(r"^\s*vs(\d+)r\.v\b", re.M)


_VEC_RELOAD = re.compile(r"^\s*vl(\d+)re?\d*\.v\b", re.M)


_CFA = re.compile(r"\.cfi_def_cfa_offset\s+(\d+)|\.cfi_escape[^#\n]*#\s*sp\s*\+"
                  r"\s*(\d+)(?:\s*\+\s*(\d+)\s*\*\s*vlenb)?")


_OBJDUMP_FIRST = re.compile(r"^[0-9a-f]+ <[^>]+>:\n\s*[0-9a-f]+:\s+([0-9a-f]+)\s", re.M)


_SPIKE_OPENS_A_WINDOW = ((0xe003, 0x6001), (0x707f, 0x0013))


def _get_stack_depth(text):
    depth = 0
    for m in _CFA.finditer(text):
        got = (int(m.group(1)) if m.group(1) else int(m.group(2))
               + int(m.group(3) or 0) * machine.VLEN_BITS // 8)
        depth = max(depth, got)
    return depth


def _get_first_instruction(ctx, stage, obj):
    out = run_tool(ctx, stage, [toolchain.RISCV_OBJDUMP, "-d", obj]).stdout
    m = _OBJDUMP_FIRST.search(out)
    return int(m.group(1), 16) if m else 0


def _is_a_window_spike_opens(bits):
    return any(bits & mask == match for mask, match in _SPIKE_OPENS_A_WINDOW)


def _check_stack_is_allowed(ctx, asm, obj, usage):
    with open(asm) as f:
        text = f.read()
    stores = [int(n) for n in _VEC_SPILL.findall(text)]
    loads = [int(n) for n in _VEC_RELOAD.findall(text)]
    depth = _get_stack_depth(text)
    fits = usage + depth <= machine.SPAD_SIZE
    if stores or loads:
        ctx.log(f"[binary] llc spilled {len(stores)} vector register group(s) "
                f"and reloads {len(loads)} at LMUL {ctx.lmul}; the stack is "
                f"{depth} bytes over {usage} bytes/lane of tiles in a "
                f"{machine.SPAD_SIZE}-byte bank")
        opens = _is_a_window_spike_opens(_get_first_instruction(ctx, "binary", obj))
        if not opens or not fits:
            ctx.log("[binary] spill not allowed: " + (
                "the first instruction opens no stack window" if not opens
                else "the stack does not fit above the tiles"))
            raise SpillNotAllowed(
                "binary",
                f"llc spilled {len(stores)} vector register group(s) "
                + ("even at LMUL 1" if ctx.lmul == 1 else f"at LMUL {ctx.lmul}"),
                artifact=obj)
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
    _check_vector_budget(ctx, ll, stage)
    t = machine.target()
    common = ["-relocation-model=pic", f"-march={t.llc_march()}", "-O3", "--stack-size-section",
              f"-mattr={t.llc_mattr()}", ll]
    run_tool(ctx, stage, [toolchain.LLC, *common, "-filetype=obj", "-o", obj])
    run_tool(ctx, stage, [toolchain.LLC, *common, "-o", asm])
    _check_no_lane_extract(ctx, asm, stage)

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
    if override:
        shutil.copyfile(override, wrap_c)
        ctx.log(f"[{tag}] using hand-written wrapper {override}")
    else:
        src = wrapper.generate(ctx.spec, t, single_program=single_program)
        with open(wrap_c, "w") as f:
            f.write(src)
        ctx.log(f"[{tag}] generated wrapper (see {os.path.basename(wrap_c)} "
                f"for the assumed ABI)")

    from ..passes import p13_lower_to_llvm as p13
    link = [toolchain.RISCV_GCC, "-O3", f"-march={t.gcc_march()}", f"-mabi={t.gcc_mabi()}",
            wrap_c, obj, "-o", elf,
            f"-Wl,--section-start={p13.SPAD_SECTION}"
            f"=0x{t.spad_vaddr:x}",
            f"-Wl,--section-start={p13.SPAD_TOP_SECTION}=0x{t.spad_vaddr + t.banked_bytes():x}"]
    run_tool(ctx, stage, link)

    usage = _check_spad_budget(ctx, elf, stage, tag)
    return elf, usage


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
        artifacts={"elf": os.path.basename(elf),
                    "post_vcix_ir": _relative_if_present(ctx, "09-custom.mlir"),
                    "llvm_ir": os.path.basename(ll),
                    "asm": os.path.basename(asm),
                    "wrapper_c": os.path.basename(wrap_c),
                    "quality": _relative_if_present(ctx, quality.FILENAME)})

    ctx.log(f"[binary] ELF: {elf}")
    ctx.log(f"[binary] kernel object: {manifest}")
    return elf
