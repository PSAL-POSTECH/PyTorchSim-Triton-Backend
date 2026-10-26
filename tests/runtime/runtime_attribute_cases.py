"""What a launch makes of a kernel's record files: the attribute files a TOG reads, and every record that
cannot be one refused by name; the C runtime's file read back, and a manifest of another format.
    python3 tests/runtime/runtime_attribute_cases.py        exit 0 iff every case gets what it names"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from pytorchsim_triton_compiler.contract import kernel_object as ko  # noqa: E402
from pytorchsim_triton_compiler.runtime import spike  # noqa: E402

HASH = 0x0123456789ABCDEF
PROGRAM = ko.RUNTIME_ATTRIBUTE_PROGRAM_ID
MANIFEST = {"runtime_attributes": {"table_hash": HASH, "attributes": [
    {"id": 0, "rank": 0, "capacity": [], "fill": None, "elem_bits": 64},
    {"id": 1, "rank": 1, "capacity": [4], "fill": None, "elem_bits": 64},
    {"id": 2, "rank": 2, "capacity": [2, 3], "fill": 1, "elem_bits": 32}]}}


def _body(extents, values, bits=64):
    dtype = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}[bits]
    return np.array(extents, dtype=np.int64).tobytes() + np.array(list(values), dtype=dtype).tobytes()


def _file(records, magic=ko.RECORD_FILE_MAGIC, version=ko.TOGSIM_VERSION_NUMBER, digest=HASH):
    """A record file as pytorchsim_runtime_attribute_write lays one out: the header, then each record."""
    out = magic + struct.pack("<IQ", version, digest)
    for key, i, body in records:
        out += struct.pack("<QII", key, i, len(body)) + body + bytes(-len(body) & 7)
    return out


def _program(key, *records):
    return [(key, PROGRAM, b"")] + [(key, i, body) for i, body in records]


def _convert(out, *files, manifest=MANIFEST):
    """`files` as the record files of workers 0, 1, ... under runtime `out` (None: no file), converted."""
    os.makedirs(os.path.join(out, ko.RUNTIME_ATTRIBUTE_DIR), exist_ok=True)
    for w, data in enumerate(files):
        if data is not None:
            open(os.path.join(out, ko.record_file(w)), "wb").write(data)
    return spike.write_runtime_attributes(manifest, out, list(range(len(files))))[0]


def _refused(out, *files, manifest=MANIFEST):
    try:
        _convert(out, *files, manifest=manifest)
    except ko.ObjectError as e:
        return str(e)
    return "no refusal"


def _attr(out, name):
    b = open(os.path.join(out, ko.RUNTIME_ATTRIBUTE_DIR, name), "rb").read()
    return list(np.frombuffer(b[:16], np.uint32)), list(np.frombuffer(b[16:], np.int64))


S7, A7 = (0, _body([], [42])), (1, _body([3], [5, 6, 7]))
G7 = [(2, _body([2, 3], [1, 2, 3], 32)), (2, _body([2, 3], [4, 5, 6], 32))]


def records_are_one_file_per_key_and_id(out):
    """Breaks if a record moves out of attr_<key>_<id>.bin, its header, its extents or its values, a G
    record's trips stop stacking in file order, programs.bin stops naming the marked keys, or a worker's
    record file is left beside them (TOGSim refuses any other file there)."""
    keys = _convert(out, _file(_program(7, S7, A7, *G7)), _file(_program(9, (1, _body([0], [])))))
    assert keys == [7, 9]
    d = os.path.join(out, ko.RUNTIME_ATTRIBUTE_DIR)
    assert sorted(os.listdir(d)) == ["attr_7_0.bin", "attr_7_1.bin", "attr_7_2.bin", "attr_9_1.bin",
                                     "programs.bin"], os.listdir(d)
    b = open(os.path.join(d, "programs.bin"), "rb").read()
    assert list(np.frombuffer(b[:8], np.uint32)) == [0x50415354, 0x10000]
    assert list(np.frombuffer(b[8:], np.uint64)) == [2, 7, 9]
    assert _attr(out, "attr_7_0.bin") == ([0x41525354, 0x10000, 0, 0], [7, 42])
    assert _attr(out, "attr_7_1.bin") == ([0x41525354, 0x10000, 1, 1], [7, 3, 5, 6, 7])
    assert _attr(out, "attr_7_2.bin") == ([0x41525354, 0x10000, 2, 2], [7, 2, 3, 1, 2, 3, 4, 5, 6])


def an_array_of_no_trips_is_present_and_empty(out):
    """Breaks if a record of extent 0 is dropped: 0 trips is not a missing record."""
    _convert(out, _file(_program(0, (1, _body([0], [])))))
    assert _attr(out, "attr_0_1.bin")[1] == [0, 0]


def a_header_of_another_magic_version_or_table_is_refused(out):
    assert "magic b'XXXX'" in _refused(out, _file(_program(0), magic=b"XXXX"))
    assert "version 0x20000" in _refused(out, _file(_program(0), version=0x20000))
    assert "another compile's ELF wrote it" in _refused(out, _file(_program(0), digest=HASH + 1))


def an_unknown_id_is_refused(out):
    assert "attribute 5, which the manifest does not declare" in _refused(out, _file(_program(0, (5, b""))))


def bytes_that_do_not_follow_the_extents_are_refused(out):
    """Breaks if a record whose byte count is not its rank's extents and the values they hold, at the
    attribute's element width, is read: S of 16 bytes, A of 3 extents holding 2, G of a short row."""
    got = [_refused(out, _file(_program(0, rec))) for rec in (
        (0, bytes(16)), (1, _body([3], [5, 6])), (2, _body([1, 3], [1, 2], 32)), (1, b""))]
    assert "16 bytes, its extents [] of 64-bit values need 8" in got[0], got[0]
    assert "its extents [3] of 64-bit values need 32" in got[1], got[1]
    assert "its extents [1, 3] of 32-bit values need 28" in got[2], got[2]
    assert "cannot hold its 1 extents" in got[3], got[3]


def two_records_of_one_s_or_a_id_are_refused(out):
    assert "recorded attribute 0 2 times" in _refused(out, _file(_program(3, S7, S7)))
    assert "recorded attribute 1 2 times" in _refused(out, _file(_program(3, A7, A7)))


def extents_past_capacity_are_refused(out):
    assert "exceed its static capacity [4]" in _refused(out, _file(_program(0, (1, _body([5], range(5))))))


def snapshot_records_of_other_extents_are_refused(out):
    got = _refused(out, _file(_program(0, G7[0], (2, _body([1, 3], [4, 5, 6], 32)))))
    assert "its records' extents differ" in got, got


def an_unfilled_snapshot_is_refused(out):
    assert "filled 1 of 2 trip rows" in _refused(out, _file(_program(0, G7[0])))


def a_record_with_no_program_marker_is_refused(out):
    """Breaks if a record is taken for a key its file has not marked: before any marker, or after the
    marker of another key."""
    assert "no program marker of its key" in _refused(out, _file([(4, 0, _body([], [1]))] + _program(4)))
    assert "no program marker of its key" in _refused(out, _file(_program(4) + [(5, 0, _body([], [1]))]))


def a_second_program_marker_is_refused(out):
    assert "a second program marker of key 3" in _refused(out, _file(_program(3)), _file(_program(3)))
    assert "carries 8 bytes" in _refused(out, _file([(3, PROGRAM, bytes(8))]))


def a_missing_or_unopened_worker_file_is_refused(out):
    assert "no record file; its worker never ran" in _refused(out, _file(_program(0)), None)
    assert "no record file header" in _refused(out, _file(_program(0)), b"")


def a_cut_record_is_refused(out):
    assert "cut at byte" in _refused(out, _file(_program(0, S7))[:-4])


def no_program_ran_is_refused(out):
    assert "no program ran" in _refused(out, _file([]))


RUNTIME_MAIN = r"""
#include <stdint.h>
const uint64_t pytorchsim_runtime_attribute_table_hash = 0x0123456789ABCDEFULL;
void pytorchsim_runtime_attribute_open(const char *path);
extern uint64_t pytorchsim_runtime_attribute_key;
void pytorchsim_runtime_attribute_write(uint32_t id, const void *data, uint64_t nbytes);
void pytorchsim_runtime_attribute_close(void);
int main(int argc, char **argv) {
  int64_t s[1] = {42}, a[4] = {3, 5, 6, 7};
  struct { int64_t e[2]; int32_t v[3]; } g = {{2, 3}, {1, 2, 3}};
  pytorchsim_runtime_attribute_open(argv[1]);
  pytorchsim_runtime_attribute_key = 7;
  pytorchsim_runtime_attribute_write(0xffffffffu, 0, 0);
  pytorchsim_runtime_attribute_write(0, s, 8);
  pytorchsim_runtime_attribute_write(1, a, 32);
  pytorchsim_runtime_attribute_write(2, &g, 28);
  g.v[0] = 4; g.v[1] = 5; g.v[2] = 6;
  pytorchsim_runtime_attribute_write(2, &g, 28);
  pytorchsim_runtime_attribute_close();
  return 0;
}
"""


def _host_runtime(work, main):
    """`main` linked with the C runtime for this host, as the wrapper links it for Spike."""
    import subprocess
    from pytorchsim_triton_compiler.lowering.elf import wrapper
    src, exe = os.path.join(work, "main.c"), os.path.join(work, "rec")
    open(src, "w").write(main)
    subprocess.run(["gcc", "-O2", *wrapper.runtime_flags(), src, wrapper.RUNTIME_SOURCE, "-o", exe], check=True)
    return exe


def the_c_runtime_writes_what_the_conversion_reads(out):
    """Breaks if runtime_attribute.c and the conversion disagree about the record file: the C runtime's
    file is byte-identical to the layout this file states, and converts to the same attribute files."""
    import subprocess
    exe = _host_runtime(os.path.dirname(out), RUNTIME_MAIN)
    os.makedirs(os.path.join(out, ko.RUNTIME_ATTRIBUTE_DIR))
    path = os.path.join(out, ko.record_file(0))
    subprocess.run([exe, path], check=True)
    assert open(path, "rb").read() == _file(_program(7, S7, A7, *G7))
    assert spike.write_runtime_attributes(MANIFEST, out, [0])[0] == [7]
    assert _attr(out, "attr_7_2.bin")[1] == [7, 2, 3, 1, 2, 3, 4, 5, 6]


def the_c_runtime_names_a_write_before_open(out):
    """Breaks if a record written with no record file open is dropped instead of stopping the run."""
    import subprocess
    exe = _host_runtime(os.path.dirname(out), "#include <stdint.h>\n"
                        "const uint64_t pytorchsim_runtime_attribute_table_hash = 0;\n"
                        "void pytorchsim_runtime_attribute_write(uint32_t, const void *, uint64_t);\n"
                        "int main(void) { pytorchsim_runtime_attribute_write(0, 0, 0); return 0; }\n")
    p = subprocess.run([exe], capture_output=True, text=True)
    assert p.returncode == 1 and "a record before the record file is open" in p.stderr, (p.returncode, p.stderr)


def a_manifest_of_format_7_is_refused(out):
    """Breaks if a kernel object from before the record files reads as this format's."""
    import json
    os.makedirs(out, exist_ok=True)
    json.dump({"format": 7}, open(os.path.join(out, ko.MANIFEST), "w"))
    try:
        ko.read(out)
    except ko.ObjectError as e:
        assert "format 7" in str(e) and f"format {ko.FORMAT}" in str(e) and ko.FORMAT == 10
        return
    raise AssertionError("no refusal")


TWO_KEYS_TOG = r"""
#include "togsim.hpp"
using namespace togsim;
struct TwoKeys : Tog<TwoKeys> {
  static constexpr Buf IN{.id = 1};
  static constexpr std::array buffers{IN};
  static constexpr Attribute OFF{.id = 0, .rank = 0};
  static constexpr std::array attributes{OFF};
  struct At { int64_t i; };
  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) dispatch<&TwoKeys::tile>(k, {.key = i, .spad = 1, .args = At{.i = i}});
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
    _convert(out, _file(_program(0, (0, _body([], [5]))) + _program(1)))
    d = os.path.join(out, ko.RUNTIME_ATTRIBUTE_DIR)
    src, so = os.path.join(os.path.dirname(out), "tog.cpp"), os.path.join(os.path.dirname(out), "tog.so")
    open(src, "w").write(TWO_KEYS_TOG)
    subprocess.run(["g++", "-std=c++20", "-shared", "-fPIC", "-O1", "-I", toolchain.TOGSIM_INCLUDE, src, "-o", so],
                   check=True)
    p = subprocess.run([tog_record.binary(), "--attributes", d, so], capture_output=True, text=True)
    assert p.returncode == 1 and f"work-item 1 (key 1) reads attribute 0, which {d} does not record" in p.stderr, \
        (p.returncode, p.stderr)


CASES = [records_are_one_file_per_key_and_id, an_array_of_no_trips_is_present_and_empty,
         a_header_of_another_magic_version_or_table_is_refused, an_unknown_id_is_refused,
         bytes_that_do_not_follow_the_extents_are_refused, two_records_of_one_s_or_a_id_are_refused,
         extents_past_capacity_are_refused, snapshot_records_of_other_extents_are_refused,
         an_unfilled_snapshot_is_refused, a_record_with_no_program_marker_is_refused,
         a_second_program_marker_is_refused, a_missing_or_unopened_worker_file_is_refused,
         a_cut_record_is_refused, no_program_ran_is_refused, the_c_runtime_writes_what_the_conversion_reads,
         the_c_runtime_names_a_write_before_open, a_manifest_of_format_7_is_refused,
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
