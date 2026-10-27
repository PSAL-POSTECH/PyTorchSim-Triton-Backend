"""Stage tog (--tog): what a timing simulator reads instead of running the program, from
09-custom.mlir -- the tile-operation-graph (TOG) TOGSim simulates (emit_tog.py -> tog.so), every
tile's compute type (tile_types.json), and the one-tile binary gem5 times (cycle_binary.py ->
09-sample, 10-sample-lowered, 11-cycle.*, cycle_bin)."""
from . import llvm_version  # noqa: F401 -- refuses an LLVM the passes are not written against


def recipe():
    return ["09-custom.mlir -> tog.cpp -> tog.so, tile_types.json",
            "09-custom.mlir -> 09-sample -> 10-sample-lowered -> 11-cycle.* -> cycle_bin"]


def run(ctx):
    from . import cycle_binary, emit_tog
    from ..driver import timer
    from .lower_to_emitc import finish_so
    pending = []
    with timer.PROF.step("emit-tog"):
        cpp, types, why = emit_tog.emit_tog(ctx.spec, ctx.workdir, pending)
    ctx.log(f"[tog] copy node: {why}")
    try:
        elf = cycle_binary.build_cycle_binary(ctx)
    except BaseException:
        for proc, _, _ in pending:
            proc.kill()
            proc.wait()
        raise
    for started in pending:
        finish_so(started)
    for made in (cpp, types, elf):
        ctx.log(f"[tog] {made}")
    return cpp
