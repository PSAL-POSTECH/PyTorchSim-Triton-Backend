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


def _launch_records(grid, want_jobs, offset):
    """_launch's recorded Spike over a kernel recording one scalar attribute: each program marks its key
    and records 10 * its key in the record file its argv names, which the launch converts."""
    import struct
    from pytorchsim_triton_compiler.runtime import spike
    m = dict(_manifest(grid), runtime_attributes={"table_hash": 5, "attributes": [
        {"id": 0, "rank": 0, "capacity": [], "fill": None, "elem_bits": 64}]})
    ran = []

    def run(cmd, cwd):
        tail = cmd[cmd.index("k.elf") + 1:]
        lo, hi = (int(tail[2]), int(tail[3])) if len(tail) > 2 else (0, grid)
        raw = os.path.join(cwd, tail[0])
        data = bytearray(open(raw, "rb").read())
        rec = b"TSRA" + struct.pack("<IQ", ko.TOGSIM_VERSION_NUMBER, 5)
        for p in range(lo, hi):
            data[offset(p)] = p + 1
            rec += struct.pack("<QII", p, ko.RUNTIME_ATTRIBUTE_PROGRAM_ID, 0)
            rec += struct.pack("<QIIq", p, 0, 8, 10 * p)
        open(raw, "wb").write(bytes(data))
        open(os.path.join(cwd, tail[1]), "wb").write(rec)
        ran.append(tail)

    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "o.raw")
        open(raw, "wb").write(bytes(grid))
        spike.launch("k.elf", m, d, {"o": raw}, OPTIONS, spike="spike", pk="pk",
                     log=lambda *_: None, run=run, want_jobs=want_jobs)
        out = os.path.join(d, ko.RUNTIME_ATTRIBUTE_DIR)
        vals = {n: int.from_bytes(open(os.path.join(out, n), "rb").read()[-8:], "little")
                for n in os.listdir(out) if n.startswith("attr_")}
        return vals, sorted(os.listdir(out)), ran


def _every_worker_record_file_is_converted():
    """Each worker writes its own record file, the launch converts them all and leaves only what a TOG
    reads; a clash reruns the grid whole into one record file."""
    split, left, ran = _launch_records(8, "4", lambda p: p)
    whole, left1, ran1 = _launch_records(8, "4", lambda p: 0)
    want = {f"attr_{p}_0.bin": 10 * p for p in range(8)}
    ok = (split == whole == want and left == left1 == sorted(want) + ["programs.bin"]
          and sorted(r[1] for r in ran) == [f"runtime_attribute/{w}.rec" for w in range(4)]
          and ran1[-1] == ["o.raw", "runtime_attribute/0.rec"])
    return ok, {"split": split, "left": left, "argv": ran, "whole_argv": ran1[-1:]}


def _inout_is_one_process():
    """An argument read as well as written keeps the grid in one process, whatever was asked."""
    from pytorchsim_triton_compiler.runtime import spike
    said = []
    n = spike.jobs(_manifest(8, role="inout"), 8, said.append, want=4)
    return n == 1 and "o is read as well as written" in said[0], {"jobs": n, "log": said}


CASES = {"float8_has_a_size": _float8_has_a_size,
         "split_matches_one_process": _split_matches_one_process,
         "a_clash_runs_the_grid_whole": _a_clash_runs_the_grid_whole,
         "every_worker_record_file_is_converted": _every_worker_record_file_is_converted,
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
