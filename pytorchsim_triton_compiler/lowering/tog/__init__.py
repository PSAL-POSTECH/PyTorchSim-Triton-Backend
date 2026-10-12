"""Stage tog (--tog): what a timing simulator reads instead of running the program, from
09-custom.mlir -- the tile-operation-graph (TOG) TOGSim replays (emit_tog.py -> tog.so), every
tile's compute type (tile_types.json), and the one-tile binary gem5 times (cycle_binary.py ->
09-sample, 10-sample-lowered, 11-cycle.*, cycle_bin)."""

GROUPINGS = ("typed", "segment")
"""How the TOG groups compute into tiles: `segment` by the DMA, wait and replayed-loop boundaries
alone; `typed` by unit and opcode, DEPRECATED and kept so the two can be compared."""

DEFAULT_GROUPING = "typed"
"""What a compile that names no grouping gets. Still the deprecated one, until the comparison is
decided."""

DEPRECATED_GROUPINGS = ("typed",)


def deprecation(grouping, asked):
    """The one-line warning for a deprecated `grouping`, or None; `asked` is whether the caller
    named it or got it as the default."""
    if grouping not in DEPRECATED_GROUPINGS:
        return None
    how = "" if asked else ", which is still the default and was not asked for here,"
    return (f"pytorchsim-triton-compiler: warning: the {grouping} TOG grouping{how} is deprecated: "
            f"it is kept for comparison and will be removed. Its replacement is --tog-grouping segment.")


def resolve(grouping):
    """(grouping, asked): `grouping`, or DEFAULT_GROUPING when the caller named none."""
    return (DEFAULT_GROUPING, False) if grouping is None else (grouping, True)


def recipe():
    return ["09-custom.mlir -> tog.cpp -> tog.so, tile_types.json",
            "09-custom.mlir -> 09-sample -> 10-sample-lowered -> 11-cycle.* -> cycle_bin"]


def run(ctx):
    import sys

    from . import cycle_binary, emit_tog
    grouping, asked = resolve(ctx.tog_grouping)
    warning = deprecation(grouping, asked)
    if warning:
        print(warning, file=sys.stderr)
        ctx.log(warning)
    cpp, types = emit_tog.emit_tog(ctx.spec, ctx.workdir, grouping)
    elf = cycle_binary.build_cycle_binary(ctx, grouping=grouping)
    for made in (cpp, types, elf):
        ctx.log(f"[tog] {made}")
    return cpp
