"""What a launch makes of a kernel's hidden output: the attribute files a TOG reads, and every row that
cannot be one refused by name; the split run's merge, and a manifest of another format.
    python3 tests/runtime/runtime_attribute_cases.py        exit 0 iff every case gets what it names"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from pytorchsim_triton_compiler.contract import kernel_object as ko  # noqa: E402

MANIFEST = {"runtime_attributes": {"file": ko.HIDDEN_FILE, "row_words": 2 + 3 + 6, "programs": 2,
                                   "attributes": [{"id": 0, "rank": 0, "offset": 2, "capacity": [], "fill": None},
                                                  {"id": 1, "rank": 1, "offset": 5, "capacity": [4],
                                                   "fill": None}]}}


def _rows(*rows):
    return np.array([list(r) + [0] * (11 - len(r)) for r in rows], dtype=np.int64)


def _write(rows, out):
    return ko.write_runtime_attributes(MANIFEST, rows, out)


def _files(out):
    got = {}
    for name in sorted(os.listdir(out)):
        b = open(os.path.join(out, name), "rb").read()
        got[name] = (list(np.frombuffer(b[:8], np.uint32)), list(np.frombuffer(b[8:], np.int64)))
    return got


def records_are_one_file_per_key_and_id(out):
    """Breaks if a record moves out of attr_<key>_<id>.bin, its header, its extents or its values, or
    programs.bin stops naming the keys that ran."""
    keys = _write(_rows([1, 7, 1, 0, 42, 1, 0, 3, 5, 6, 7], [1, 9, 0, 0, 0, 1, 0, 0]), out)
    assert keys == [7, 9]
    got = _files(out)
    assert got["programs.bin"] == ([0x50415354, 0x10000], [2, 7, 9]), got["programs.bin"]
    b = open(os.path.join(out, "attr_7_1.bin"), "rb").read()
    assert list(np.frombuffer(b[:16], np.uint32)) == [0x41525354, 0x10000, 1, 1]
    assert list(np.frombuffer(b[16:], np.int64)) == [7, 3, 5, 6, 7]
    assert "attr_9_0.bin" not in got and "attr_9_1.bin" in got and "attr_7_0.bin" in got


def an_array_of_no_trips_is_present_and_empty(out):
    """Breaks if a record of extent 0 is dropped: 0 trips is not a missing record."""
    _write(_rows([1, 0, 0, 0, 0, 1, 0, 0]), out)
    b = open(os.path.join(out, "attr_0_1.bin"), "rb").read()
    assert list(np.frombuffer(b[16:], np.int64)) == [0, 0]


def _refused(rows, out):
    try:
        _write(rows, out)
    except ko.ObjectError as e:
        return str(e)
    return "no refusal"


def two_records_of_one_id_are_refused(out):
    assert "recorded attribute 0 2 times" in _refused(_rows([1, 3, 2, 0, 5]), out)


def extents_past_capacity_are_refused(out):
    assert "exceed its static capacity [4]" in _refused(_rows([1, 3, 0, 0, 0, 1, 0, 5]), out)


def a_row_naming_a_key_twice_is_refused(out):
    assert "both name key 3" in _refused(_rows([1, 3], [1, 3]), out)


def a_row_with_another_ran_is_refused(out):
    assert "has ran=2" in _refused(_rows([2, 3]), out)


def no_program_ran_is_refused(out):
    assert "no program ran" in _refused(_rows([0], [0]), out)


def an_unfilled_snapshot_is_refused(out):
    m = {"runtime_attributes": dict(MANIFEST["runtime_attributes"], attributes=[
        {"id": 0, "rank": 2, "offset": 2, "capacity": [2, 2], "fill": 1}])}
    try:
        ko.write_runtime_attributes(m, _rows([1, 4, 1, 1, 2, 2]), out)
    except ko.ObjectError as e:
        assert "filled 1 of 2 trip rows" in str(e), e
        return
    raise AssertionError("no refusal")


def a_run_that_wrote_nothing_is_refused(out):
    path = os.path.join(out, ko.HIDDEN_FILE)
    os.makedirs(out, exist_ok=True)
    open(path, "wb").close()
    try:
        ko.read_rows(MANIFEST, path)
    except ko.ObjectError as e:
        assert "its run wrote no" in str(e), e
        return
    raise AssertionError("no refusal")


def a_hidden_output_of_another_size_is_refused(out):
    path = os.path.join(out, ko.HIDDEN_FILE)
    os.makedirs(out, exist_ok=True)
    np.zeros(5, np.int64).tofile(path)
    try:
        ko.read_rows(MANIFEST, path)
    except ko.ObjectError as e:
        assert "the manifest says 2 rows of 11 words" in str(e), e
        return
    raise AssertionError("no refusal")


def the_split_merge_takes_each_row_from_the_worker_that_ran_it(out):
    """Breaks if the merge infers a row from its bytes rather than its ran word, or lets two ranges
    claim one program."""
    a, b = _rows([1, 0, 1, 0, 0], [0]), _rows([0], [1, 1, 1, 0, 0])
    assert ko.merge_rows(MANIFEST, [a, b]).tolist() == _rows([1, 0, 1], [1, 1, 1]).tolist()
    try:
        ko.merge_rows(MANIFEST, [a, a])
    except ko.ObjectError as e:
        assert "ran in two program ranges" in str(e)
        return
    raise AssertionError("no refusal")


def a_manifest_of_format_6_is_refused(out):
    """Breaks if a kernel object from before the hidden output reads as this format's."""
    import json
    os.makedirs(out, exist_ok=True)
    json.dump({"format": 6}, open(os.path.join(out, ko.MANIFEST), "w"))
    try:
        ko.read(out)
    except ko.ObjectError as e:
        assert "format 6" in str(e) and f"format {ko.FORMAT}" in str(e) and ko.FORMAT == 7
        return
    raise AssertionError("no refusal")


TWO_KEYS_TOG = r"""
#include "togsim.hpp"
using namespace togsim;
struct TwoKeys : Tog<TwoKeys> {
  static constexpr Buf IN{.id = 1, .bytes = f32(16)};
  static constexpr std::array buffers{IN};
  static constexpr Attribute OFF{.id = 0, .rank = 0};
  static constexpr std::array attributes{OFF};
  struct At { int64_t i; };
  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) dispatch<&TwoKeys::tile>(k, {.key = i, .args = At{.i = i}});
  }
  void tile(Item& w, const At&) const {
    load(w, {.src = {.ptr = 0, .type = f32, .offset = scalar(w, OFF), .dims = {16}, .strides = {1}}, .dst = IN});
  }
};
"""


def a_missing_record_names_the_work_item_as_togsim_does(out):
    """Breaks if tog_record's E11 stops naming the work-item by its dispatch index, TOGSim's
    `work-item <n> (key <k>)`: the two readers of one directory would say different things."""
    import subprocess
    from pytorchsim_triton_compiler import toolchain
    from tests.runtime import tog_record
    _write(_rows([1, 0, 1, 0, 5], [1, 1, 1, 0, 9]), out)
    os.remove(os.path.join(out, "attr_1_0.bin"))
    src, so = os.path.join(os.path.dirname(out), "tog.cpp"), os.path.join(os.path.dirname(out), "tog.so")
    open(src, "w").write(TWO_KEYS_TOG)
    subprocess.run(["g++", "-std=c++20", "-shared", "-fPIC", "-O1", "-I", toolchain.TOGSIM_INCLUDE, src, "-o", so],
                   check=True)
    p = subprocess.run([tog_record.binary(), "--attributes", out, so], capture_output=True, text=True)
    assert p.returncode == 1 and f"work-item 1 (key 1) reads attribute 0, which {out} does not record" in p.stderr, \
        (p.returncode, p.stderr)


CASES = [records_are_one_file_per_key_and_id, an_array_of_no_trips_is_present_and_empty,
         two_records_of_one_id_are_refused, extents_past_capacity_are_refused,
         a_row_naming_a_key_twice_is_refused, a_row_with_another_ran_is_refused, no_program_ran_is_refused,
         an_unfilled_snapshot_is_refused, a_run_that_wrote_nothing_is_refused,
         a_hidden_output_of_another_size_is_refused,
         the_split_merge_takes_each_row_from_the_worker_that_ran_it, a_manifest_of_format_6_is_refused,
         a_missing_record_names_the_work_item_as_togsim_does]


def main():
    failed = 0
    for case in CASES:
        with tempfile.TemporaryDirectory() as d:
            try:
                case(os.path.join(d, "out"))
                print(f"PASS  {case.__name__}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
