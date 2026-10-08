"""Everything a timing simulator reads instead of running the program."""

GROUPINGS = ("typed", "segment")
"""How the TOG groups compute into tiles: `typed` by unit and opcode, `segment` by the DMA, wait
and replayed-loop boundaries alone."""
