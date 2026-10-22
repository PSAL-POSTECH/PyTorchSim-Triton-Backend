"""Async loads on two semaphores -- one waited after its loop, one inside it -- lowered to a TOG, built
against togsim.hpp and run: one togsim::semaphore per slot, each wait the elements its loads
signalled, and a barrier on a slot no async load signals refused.
    python3 tests/runtime/semaphore_cases.py        exit 0 iff every case gets what it names"""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytorchsim_triton_compiler  # noqa: E402,F401
from pytorchsim_triton_compiler.lowering.tog.passes import lower_to_emitc as l2e  # noqa: E402
from pytorchsim_triton_compiler.lowering.tog.passes.build_tog import ir  # noqa: E402
from runtime import tog_replay  # noqa: E402

LOAD = """"torchsim.dma"(%i, {sem}) {{dir = 0 : i32, arg_id = 0 : i32, dims = array<i64: {n}>,
          strides = array<i64: 1>, elem = "f32", is_async = true, read_bufs = array<i64>,
          write_bufs = array<i64: {buf}>, indirect = 0 : i32}} : (index, index) -> ()"""
SKELETON = """
module {
  func.func @kernel(%x: memref<*xf32>, %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32,
                    %pz: i32) attributes {togsim.spad_buffer_bytes = array<i64: 512, 512, 1024>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %s0 = arith.constant 0 : index
    %s1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      """ + LOAD.format(sem="%s0", n=128, buf=0) + """
      """ + LOAD.format(sem="%s1", n=64, buf=1) + """
      "torchsim.memory_barrier"(%s1) : (index) -> ()
    }
    "torchsim.memory_barrier"(%s0) : (index) -> ()
    "torchsim.compute"() {tile_id = 0 : i64, compute_type = 0 : i32, read_bufs = array<i64: 0, 1>,
        write_bufs = array<i64: 2>} : () -> ()
    "torchsim.dma"(%c0, %s0) {dir = 1 : i32, arg_id = 0 : i32, dims = array<i64: 256>,
        strides = array<i64: 1>, elem = "f32", is_async = false, read_bufs = array<i64: 2>,
        write_bufs = array<i64>, indirect = 0 : i32} : (index, index) -> ()
    return
  }
}
"""


def _cpp(text=SKELETON):
    """The TOG's C++ for `text` on a grid of one axis, or the refusal."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        item = l2e.WorkItem(parallel_args=[4], grid=[None], key_shifts=[0], grid_args=[1],
                            fixed_args={2: 1, 3: 1}, kernel="probe", axes=["x"], example=[2])
        try:
            return l2e.emitc_to_cpp(l2e.lower_to_emitc(ir.Module.parse(text), work_item=item))
        except ValueError as e:
            return str(e)


def _semaphores():
    return [re.sub(r"v\d+", "v", l.strip()) for l in _cpp().splitlines() if "emaphore" in l]


def _barriers():
    """Each work-item's barriers as (sem, expected), run on a grid of 2."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "tog.cpp"), "w").write(_cpp())
        so = l2e.compile_so(os.path.join(d, "tog.cpp"), os.path.join(d, "tog.so"))
        proc = subprocess.run([tog_replay.binary(), so, "2"], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip())
    items = []
    for e in map(json.loads, proc.stdout.splitlines()):
        if e["op"] == "dispatch":
            items.append([])
        elif e["op"] == "barrier":
            items[-1].append((e["sem"], e["expected"]))
    return items


CASES = {
    "one semaphore per slot": (_semaphores, ["togsim::Semaphore v = togsim::semaphore(v);"] * 2),
    "waits counted": (_barriers, [[(2, 64)] * 4 + [(1, 512)]] * 2),
    "unsignalled barrier refused": (lambda: _cpp(SKELETON.replace('barrier"(%s1)', 'barrier"(%c4)')),
                                    "a barrier on semaphore 4, which no async load signals"),
}


def main():
    failed = 0
    for name, (case, want) in CASES.items():
        try:
            got = case()
        except Exception as e:  # noqa: BLE001
            got = f"{type(e).__name__}: {e}"
        ok = (want in got) if isinstance(want, str) and isinstance(got, str) else got == want
        failed += not ok
        print(f"{name:32} {'PASS' if ok else 'FAIL'}  got {str(got)[:110]}  want {str(want)[:60]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
