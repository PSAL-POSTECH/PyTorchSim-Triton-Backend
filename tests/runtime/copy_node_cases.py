"""ops_resident_matmul built and timed: the object lists B's copy node (one whole load), the kernel's TOG
reads B held, togcheck takes that only as declared, and the timing stage and simulate_shapes count the copy
node beside the kernel; stated graph_prefetch () holds nothing, (0,) (A, streamed) stops.
python3 tests/runtime/copy_node_cases.py        exit 0 iff every case holds"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "validation", "gemm"))
from runtime import tog_record  # noqa: E402
from pytorchsim_triton_compiler.contract import kernel_object as ko  # noqa: E402

CASE = os.path.join(ROOT, "tests", "cases", "mechanism", "ops", "ops_resident_matmul.py")
B, B_BYTES = 1, 256 * 192 * 2


def _built(wd):
    """The case compiled through the timing stage into `wd`; its manifest."""
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), CASE, "--workdir", wd,
                        "--tog", "--timing"], cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(f"the build failed: {(p.stdout + p.stderr)[-600:]}")
    return ko.read(wd)


def the_object_lists_one_copy_node_of_b(wd, manifest):
    nodes = ko.copy_nodes(manifest)
    assert [(n["arg"], n["bytes"]) for n in nodes] == [(B, B_BYTES)], nodes


def the_copy_node_states_b_over_the_lanes_and_the_kernel_its_elf(wd, manifest):
    """Breaks if a copy node or the kernel it holds an operand for states a spad TOGSim must infer: the copy
    node B's bytes over the lanes, the kernel its ELF's .spad, which holds B."""
    from pytorchsim_triton_compiler import toolchain
    from pytorchsim_triton_compiler.lowering.elf.build import _measure_spad
    node, lanes = ko.copy_nodes(manifest)[0], manifest["target"]["vpu_num_lanes"]
    copy = {e["spad"] for e in tog_record.record_copy_node(wd, node) if e["op"] == "dispatch"}
    kernel = {e["spad"] for e in tog_record.record(wd, ko.shape_args(manifest)) if e["op"] == "dispatch"}
    elf = _measure_spad(ko.artifact(wd, "elf"), toolchain.RISCV_READELF)
    assert copy == {-(-B_BYTES // lanes)} == {node[ko.SPAD]}, (copy, node)
    assert kernel == {elf} and elf >= -(-B_BYTES // lanes), (kernel, elf)


def the_copy_node_is_one_load_of_b_whole(wd, manifest):
    events = tog_record.record_copy_node(wd, ko.copy_nodes(manifest)[0])
    got = [(e["op"], e.get("dir"), e.get("arg"), e.get("offset"), e.get("dims"), e.get("strides")) for e in events
           if e["op"] != "attributes"]
    assert got == [("dispatch",) + (None,) * 5, ("dma", 0, B, 0, [B_BYTES // 2], [1])], got


def the_kernel_loads_no_b_and_reads_its_held_buffer(wd, manifest):
    held = ko.copy_nodes(manifest)[0]["held_buffer"]
    events = tog_record.record(wd, ko.shape_args(manifest))
    loads = [e for e in events if e["op"] == "dma" and e["dir"] == 0]
    assert loads and all(e["arg"] != B for e in loads), loads
    assert any(held in e["read"] for e in events if e["op"] == "compute"), held
    assert not any(held in e.get("write", []) for e in events), held


def each_reload_waits_for_the_compute_that_read_its_buffer(wd, manifest):
    """Breaks if the streamed loads of a kernel that holds an operand run ahead of their readers: in a
    16384x4096x4096 resident mm they overtake the one buffer each the ELF has for them.
    """
    events = tog_record.record(wd, ko.shape_args(manifest))
    loads = [e for e in events if e["op"] == "dma" and e["dir"] == 0]
    for e in loads:
        token = [b for b in e["read"] if b not in e["write"]]
        assert len(token) == 1, e
        assert any(token[0] in c["write"] and set(e["write"]) & set(c["read"]) for c in events if c["op"] == "compute"), e


def each_compute_updates_what_it_writes_in_place(wd, manifest):
    """Breaks if a compute of a kernel that holds an operand writes a new version of its output tile: in a
    16384x4096x4096 resident mm the 16 pops of one tile pile 16 versions of it, where the ELF has one."""
    events = tog_record.record(wd, ko.shape_args(manifest))
    tokens = {b for e in events if e["op"] == "dma" and e["dir"] == 0 for b in e["read"] if b not in e["write"]}
    computes = [e for e in events if e["op"] == "compute"]
    assert computes and all(set(c["write"]) - tokens <= set(c["read"]) for c in computes), computes[:2]


def togcheck_takes_the_held_read_as_declared(wd, manifest):
    problems, _ = tog_record.check(wd)
    assert problems == [], problems


def togcheck_refuses_the_held_read_undeclared(wd, manifest):
    with tempfile.TemporaryDirectory() as d:
        for name in os.listdir(wd):
            src = os.path.join(wd, name)
            if os.path.isfile(src):
                os.symlink(src, os.path.join(d, name))
        os.remove(os.path.join(d, ko.MANIFEST))
        with open(os.path.join(d, ko.MANIFEST), "w") as fh:
            json.dump({k: v for k, v in manifest.items() if k != ko.COPY_NODES}, fh)
        problems, _ = tog_record.check(d)
    held = ko.copy_nodes(manifest)[0]["held_buffer"]
    assert any(f"reads buffer {held}, which nothing" in p for p in problems), problems
    assert any("never emits: dir=0 arg=1" in p for p in problems), problems


def the_elf_keeps_the_one_load_the_copy_node_times(wd, manifest):
    events = [t for t in tog_record.ir_transfers(wd) if t["arg"] == B]
    assert len(events) == 1 and events[0]["dir"] == 0, events


def the_timing_stage_counts_the_copy_node(wd, manifest):
    got = json.load(open(os.path.join(wd, "togsim.json")))
    copies = got["copy_nodes"]
    assert len(copies) == 1 and copies[0]["total_cycles"] > 0, copies
    assert got["total_cycles"] == got["kernel_cycles"] + copies[0]["total_cycles"], got


def simulate_shapes_reads_the_launch_and_its_parts(wd, manifest):
    import simulate_shapes as ss
    tog = json.load(open(os.path.join(wd, "togsim.json")))
    one = {}
    ss.read_togsim(one, wd, {k: None for k in ss.RAN_ON} | {"core_freq_mhz": 1})
    assert (one["total_cycles"], one["kernel_cycles"], one["copy_cycles"]) == (
        tog["total_cycles"], tog["kernel_cycles"], tog["copy_nodes"][0]["total_cycles"]), one


def the_timing_key_reads_the_copy_node(wd, manifest):
    import simulate_shapes as ss
    machine = {"timing_tools": "t"}
    before = ss.timing_key(wd, machine)
    so = os.path.join(wd, ko.copy_nodes(manifest)[0]["tog_so"])
    saved = so + ".saved"
    shutil.copyfile(so, saved)
    try:
        with open(so, "ab") as fh:
            fh.write(b"\0")
        assert ss.timing_key(wd, machine) != before
    finally:
        os.replace(saved, so)


CASES = [the_object_lists_one_copy_node_of_b, the_copy_node_states_b_over_the_lanes_and_the_kernel_its_elf,
         the_copy_node_is_one_load_of_b_whole,
         the_kernel_loads_no_b_and_reads_its_held_buffer, each_reload_waits_for_the_compute_that_read_its_buffer,
         each_compute_updates_what_it_writes_in_place, togcheck_takes_the_held_read_as_declared,
         togcheck_refuses_the_held_read_undeclared, the_elf_keeps_the_one_load_the_copy_node_times, the_timing_stage_counts_the_copy_node,
         simulate_shapes_reads_the_launch_and_its_parts, the_timing_key_reads_the_copy_node]


def _built_stating(d, graph_prefetch):
    """The case with KernelSpec.graph_prefetch set, compiled with --tog in `d`: (exit code, workdir)."""
    src = os.path.join(d, f"mm_{len(graph_prefetch)}.py")
    with open(CASE) as fh, open(src, "w") as out:
        out.write(fh.read() + "\nimport dataclasses\n"
                  f"SPEC = dataclasses.replace(SPEC, graph_prefetch={graph_prefetch!r})\n")
    wd = src[:-3]
    driver = os.path.join(ROOT, "tests", "pytorchsim-compile")
    p = subprocess.run([sys.executable, driver, src, "--workdir", wd, "--tog"], cwd=ROOT, capture_output=True,
                       text=True)
    return p.returncode, wd


def a_graph_prefetching_elsewhere_builds_no_copy_node(d):
    """Breaks if graph_prefetch () is ignored: the kernel would hold B as well as the graph's prefetch."""
    rc, wd = _built_stating(d, ())
    assert rc == 0, rc
    manifest = ko.read(wd)
    assert ko.copy_nodes(manifest) == [] and manifest["graph_prefetch"] == [], manifest


def a_graph_prefetch_of_no_candidate_stops_with_its_failure_kind(d):
    """Breaks if holding A, which the kernel streams, falls back to B or to nothing instead of stopping with
    the failure.json kind the front end reads."""
    from pytorchsim_triton_compiler.contract import failure
    rc, wd = _built_stating(d, (0,))
    got = json.load(open(os.path.join(wd, failure.FILE)))
    assert rc != 0 and got["kind"] == failure.NOT_A_PREFETCH_CANDIDATE, (rc, got)


def a_copy_node_beside_runtime_attributes_is_recorded_without_them(d):
    """Breaks if togcheck records a copy node with the kernel's runtime attribute records:
    chunked_prefill_attn holds an operand and reads 4 records, which the copy node's TOG, one load, declares
    none of."""
    wd = os.path.join(d, "attn")
    case = os.path.join(ROOT, "tests", "cases", "workloads", "chunked_prefill_attn.py")
    driver = os.path.join(ROOT, "tests", "pytorchsim-compile")
    p = subprocess.run([sys.executable, driver, case, "--workdir", wd, "--tog"], cwd=ROOT,
                       capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout + p.stderr)[-600:]
    manifest = ko.read(wd)
    records = tog_record.attributes_of(os.path.join(wd, "runtime"))
    assert ko.copy_nodes(manifest) and os.listdir(records), manifest


STATED = [a_copy_node_beside_runtime_attributes_is_recorded_without_them,
          a_graph_prefetching_elsewhere_builds_no_copy_node,
          a_graph_prefetch_of_no_candidate_stops_with_its_failure_kind]


def main():
    failed = 0
    with tempfile.TemporaryDirectory() as d:
        wd = os.path.join(d, "mm")
        manifest = _built(wd)
        runs = [(case, (wd, manifest)) for case in CASES] + [(case, (d,)) for case in STATED]
        for case, args in runs:
            try:
                case(*args)
                print(f"PASS  {case.__name__}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
