"""tritonpytorchsim-lower-cross-lane-tpuv6e emits a permute's rows as the unit takes them: its bank
reads and writes are in bounds, read off the IR right after the pass.
    python3 tests/passes/xlu_permute_emission_cases.py   exit 0 iff all hold"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite_target  # noqa: E402
os.environ.setdefault("TORCHSIM_COMPILE_TARGET", suite_target.NAME)
from pytorchsim_triton_compiler import toolchain  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e import lib_xlu  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.vcix_tpuv6e.lower_cross_lane import (  # noqa: E402
    _SF_VC_IVV_SE)
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

GQA = os.path.join(ROOT, "tests", "cases", "workloads", "gqa_repeat_kv.py")
RETILE = os.path.join(ROOT, "tests", "cases", "mechanism", "tile", "tile_retile_div_rem.py")
TRANSFERS = ("vector.transfer_read", "vector.transfer_write")
_DUMPS = {}


def _after_cross_lane(kernel):
    """The text of the IR right after the cross-lane pass, compiled once per kernel."""
    if kernel not in _DUMPS:
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "w")
            subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), kernel,
                            "--workdir", w, "--to", "lower"], check=False, capture_output=True)
            out = [f for f in sorted(os.listdir(os.path.join(w, "09-passes")))
                   if "lower-cross-lane" in f]
            _DUMPS[kernel] = open(os.path.join(w, "09-passes", out[0])).read()
    return _DUMPS[kernel]


def _is_push(o):
    return (o.name == "llvm.call_intrinsic"
            and ir.StringAttr(o.attributes["intrin"]).value == _SF_VC_IVV_SE)


def _is_pop(o):
    return (o.name == "vcix.v.i"
            and ir.IntegerAttr(o.attributes["opcode"]).value == lib_xlu.XLU_POP_OPCODE)


def _unit_blocks(m):
    """Each op that directly holds an XLU push or pop: (the ops beside them, whether it pops)."""
    held = []
    for o in walk(m.operation):
        if _is_push(o) or _is_pop(o):
            p = o.parent
            for i, (q, ops, pops) in enumerate(held):
                if q == p:
                    held[i] = (q, ops, pops or _is_pop(o))
                    break
            else:
                held.append((p, [x for r in p.regions for b in r.blocks for x in b.operations],
                             _is_pop(o)))
    return [(ops, pops) for _, ops, pops in held]


def _in_bounds(o):
    return [ir.BoolAttr(x).value for x in ir.ArrayAttr(o.attributes["in_bounds"])]


def every_permute_transfer_is_in_bounds():
    """The rounds' rows are checked inside the banks before any is emitted, so no transfer of a
    push or a pop carries a bounds check. Breaks if the permute drops `in_bounds`."""
    for kernel in (GQA, RETILE):
        with ir.Context() as ctx, ir.Location.unknown():
            ctx.allow_unregistered_dialects = True
            m = ir.Module.parse(_after_cross_lane(kernel))
            got = [_in_bounds(o) for ops, _ in _unit_blocks(m) for o in ops if o.name in TRANSFERS]
            assert got and all(all(b) for b in got), (os.path.basename(kernel), got)


CASES = [every_permute_transfer_is_in_bounds]


def main():
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
