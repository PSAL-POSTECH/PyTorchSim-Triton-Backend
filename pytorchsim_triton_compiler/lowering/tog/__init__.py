"""Stage tog (--tog): what a timing simulator reads instead of running the program, from
09-custom.mlir -- the tile-operation-graph (TOG) TOGSim simulates (emit_tog.py -> tog.so), every
tile's compute type (tile_types.json), and the one-tile binary gem5 times (cycle_binary.py ->
09-sample, 10-sample-lowered, 11-cycle.*, cycle_bin)."""


def recipe():
    return ["09-custom.mlir -> tog.cpp -> tog.so, tile_types.json",
            "09-custom.mlir -> 09-sample -> 10-sample-lowered -> 11-cycle.* -> cycle_bin"]


def run(ctx):
    from . import cycle_binary, emit_tog
    cpp, types, why = emit_tog.emit_tog(ctx.spec, ctx.workdir)
    ctx.log(f"[tog] copy node: {why}")
    elf = cycle_binary.build_cycle_binary(ctx)
    for made in (cpp, types, elf):
        ctx.log(f"[tog] {made}")
    return cpp
