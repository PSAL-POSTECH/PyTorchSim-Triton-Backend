"""The package's Spike launcher (runtime/spike.py), driven with a recorded process in place of Spike,
and the dtype tables it sizes .raw files by.
    python3 tests/runtime/spike_launch_cases.py        exit 0 iff every case holds"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler.contract import kernel_object as ko  # noqa: E402

OPTIONS = {"spike_isa": "rv64gcv", "model": "m.so", "memory": "0x0:0x1", "machine_config": "m.yml"}


def _float8_has_a_size():
    """A float8 argument is one byte an element, carried as uint8, as PyTorchSim's launch writes it."""
    got = {d: (ko.nbytes({"dtype": d, "shape": [3, 4]}), ko.NP_DTYPE[d])
           for d in ("float8_e5m2", "float8_e4m3fn")}
    return all(v == (12, "uint8") for v in got.values()), got


def _manifest(grid, role="out"):
    return {"name": "k", "grid": [grid], "scalars": [], "abi": {"kernel_addr": [0, 4]},
            "args": [{"name": "o", "role": role, "dtype": "uint8", "shape": [grid]}]}


def _launch(grid, want_jobs, offset):
    """Launch a recorded Spike whose program p stores p+1 at byte offset(p) of o.raw, in the cwd-relative
    path its argv names; the bytes left in o.raw and the argv tails each process was given."""
    from pytorchsim_triton_compiler.runtime import spike
    ran = []

    def run(cmd, cwd):
        lo, hi = (int(cmd[-2]), int(cmd[-1])) if cmd[-1].isdigit() else (0, grid)
        raw = os.path.join(cwd, cmd[-3] if cmd[-1].isdigit() else cmd[-1])
        data = bytearray(open(raw, "rb").read())
        for p in range(lo, hi):
            data[offset(p)] = p + 1
        open(raw, "wb").write(bytes(data))
        ran.append(cmd[cmd.index("k.elf") + 1:])

    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "o.raw")
        open(raw, "wb").write(bytes(grid))
        spike.launch("k.elf", _manifest(grid), d, {"o": raw}, OPTIONS, spike="spike", pk="pk",
                     log=lambda *_: None, run=run, want_jobs=want_jobs)
        return list(open(raw, "rb").read()), ran, sorted(os.listdir(d))


def _split_matches_one_process():
    """Four program ranges, each storing only its own bytes, merge to what one process writes."""
    whole, ran1, _ = _launch(8, "1", lambda p: p)
    split, ran4, left = _launch(8, "4", lambda p: p)
    ok = (whole == split == list(range(1, 9)) and ran1 == [["o.raw"]]
          and sorted(ran4) == [[f"part{w}/o.raw", str(2 * w), str(2 * w + 2)] for w in range(4)]
          and "part0" not in left)
    return ok, {"whole": whole, "split": split, "argv": sorted(ran4)}


def _a_clash_runs_the_grid_whole():
    """Ranges storing one byte differently are not merged: the grid runs again in one process."""
    got, ran, _ = _launch(8, "4", lambda p: 0)
    return got[0] == 8 and len(ran) == 5 and ran[-1] == ["o.raw"], {"o": got, "argv": ran}


def _inout_is_one_process():
    """An argument read as well as written keeps the grid in one process, whatever was asked."""
    from pytorchsim_triton_compiler.runtime import spike
    said = []
    n = spike.jobs(_manifest(8, role="inout"), 8, said.append, want=4)
    return n == 1 and "o is read as well as written" in said[0], {"jobs": n, "log": said}


CASES = {"float8_has_a_size": _float8_has_a_size,
         "split_matches_one_process": _split_matches_one_process,
         "a_clash_runs_the_grid_whole": _a_clash_runs_the_grid_whole,
         "inout_is_one_process": _inout_is_one_process}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:34} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
