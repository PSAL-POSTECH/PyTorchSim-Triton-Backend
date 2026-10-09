"""Stage tog (--tog): what a timing simulator reads instead of running the program, from
09-custom.mlir -- the trace producer TOGSim runs (trace.py -> trace.so), every tile's compute type
(tile_types.json), and the one-tile binary gem5 times (cycle_binary.py -> 09-sample, 10-sample-lowered,
11-cycle.*, cycle_bin)."""

GROUPINGS = ("typed", "segment")
"""How the TOG groups compute into tiles: `typed` by unit and opcode, `segment` by the DMA, wait
and replayed-loop boundaries alone."""


def recipe():
    return ["09-custom.mlir -> trace.cpp -> trace.so, tile_types.json",
            "09-custom.mlir -> 09-sample -> 10-sample-lowered -> 11-cycle.* -> cycle_bin"]


def run(ctx):
    from . import cycle_binary, trace
    cpp, types = trace.build_trace(ctx.spec, ctx.workdir, ctx.tog_grouping)
    elf = cycle_binary.build_cycle_binary(ctx, grouping=ctx.tog_grouping)
    for made in (cpp, types, elf):
        ctx.log(f"[trace] {made}")
    return cpp
