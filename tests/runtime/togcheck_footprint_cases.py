"""togcheck's footprint verdicts at their edges -- a zero extent, an index past either end, two dmas on one
element, an element nothing stores, a uint64 offset, an extent past the cap -- and 2^24 elements checked in seconds.
    python3 tests/runtime/togcheck_footprint_cases.py        exit 0 iff every case gets what it names"""
import json
import os
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import tog_replay  # noqa: E402

NEVER_MOVED = "arg 0 (y) is never moved by any dma, so the simulator sees no traffic for it"


def _dma(offset, dims, strides, dir=1):
    return {"op": "dma", "dir": dir, "arg": 0, "offset": offset, "dims": dims, "strides": strides,
            "elem_bits": 32, "async": 0, "sem": 0, "indirect": False}


def _check(dmas, numel, role="out"):
    """check() of one work item moving `dmas` on one fp32 arg of `numel` elements; the IR names the same
    transfers, so only the footprint verdicts can speak."""
    events = [{"op": "dispatch", "key": 0, "iv": [0]}] + dmas
    transfers = [{"dir": e["dir"], "arg": 0, "dims": tuple(e["dims"]), "strides": tuple(e["strides"]),
                  "elem_bits": 32, "masked": False, "clamped_axes": (), "indirect": False, "cut": False}
                 for e in dmas]
    saved = tog_replay.replay, tog_replay.ir_transfers, tog_replay.ir_scalar_dram
    tog_replay.replay = lambda workdir, shape, timeout=600: events
    tog_replay.ir_transfers = lambda workdir: transfers
    tog_replay.ir_scalar_dram = lambda workdir: {}
    try:
        with tempfile.TemporaryDirectory() as d:
            json.dump({"args": [{"name": "y", "dtype": "fp32", "role": role, "shape": [numel]}],
                       "grid": [1], "abi": {"parallel_axes": ["x"]}}, open(os.path.join(d, "kernel.json"), "w"))
            json.dump({"compute_types": [], "dispatch_extents": []}, open(os.path.join(d, "tile_types.json"), "w"))
            return tog_replay.check(d)
    finally:
        tog_replay.replay, tog_replay.ir_transfers, tog_replay.ir_scalar_dram = saved


def _bounded():
    """Sixteen stores of 2^20 elements cover a 2^24-element output: a pass, within 10 s and 256 MiB traced."""
    rows = [_dma(r << 20, [1024, 1024], [1024, 1]) for r in range(16)]
    tracemalloc.start()
    t = time.time()
    got = _check(rows, 1 << 24)
    took, peak = time.time() - t, tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return got == ([], []) and took < 10 and peak < 256 << 20, (got, f"{took:.1f}s", f"{peak >> 20}MiB")


CASES = {
    "zero extent moves nothing": (lambda: _check([_dma(0, [0, 4], [4, 1])], 16), ([], [NEVER_MOVED])),
    "no dims is one element": (lambda: _check([_dma(3, [], [])], 4),
                               ([], ["arg 0 (y) is `out` and 3 of its 4 elements are never stored"])),
    "past the end": (lambda: _check([_dma(12, [8], [1])], 16),
                     (["arg 0 (y) has 16 elements and a dma reaches 16..19"],
                      ["arg 0 (y) is `out` and 12 of its 16 elements are never stored"])),
    "before the start": (lambda: _check([_dma(2, [4], [-1], dir=0)], 16, role="in"),
                         (["arg 0 (y) has 16 elements and a dma reaches -1..-1"], [])),
    "both ends name the extremes": (lambda: _check([_dma(2, [3, 2], [-2, 20])], 16),
                                    (["arg 0 (y) has 16 elements and a dma reaches -2..22"],
                                     ["arg 0 (y) is `out` and 14 of its 16 elements are never stored"])),
    "uint64 offset stays exact": (lambda: _check([_dma((1 << 64) - 1, [2], [1])], 16),
                                  (["arg 0 (y) has 16 elements and a dma reaches "
                                    "18446744073709551615..18446744073709551616"],
                                   ["arg 0 (y) is `out` and 16 of its 16 elements are never stored"])),
    "overlap covers once": (lambda: _check([_dma(0, [10], [1]), _dma(6, [10], [1])], 16), ([], [])),
    "overlap leaves a gap": (lambda: _check([_dma(0, [8], [1]), _dma(0, [8], [1])], 16),
                             ([], ["arg 0 (y) is `out` and 8 of its 16 elements are never stored"])),
    "strided rows leave holes": (lambda: _check([_dma(0, [2, 4], [8, 1])], 16),
                                 ([], ["arg 0 (y) is `out` and 8 of its 16 elements are never stored"])),
    "extent past the strides adds none": (lambda: _check([_dma(0, [4, 3], [1])], 16),
                                          ([], ["arg 0 (y) is `out` and 12 of its 16 elements are never stored"])),
    "past the cap is unjudged": (lambda: _check([_dma(0, [4097, 1024], [1024, 1]), _dma(0, [0], [1])], 16),
                                 ([], [])),
    "2^24 elements, bounded": (_bounded, True),
}


def main():
    failed = 0
    for name, (case, want) in CASES.items():
        try:
            got = case()
        except Exception as e:
            got = f"{type(e).__name__}: {e}"
        ok, shown = (got[0], got[1]) if want is True and isinstance(got, tuple) else (got == want, got)
        failed += not ok
        print(f"{name:36} {'PASS' if ok else 'FAIL'}  {str(shown)[:160]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
