"""lower_cross_lane pushes every round of a permute before its first pop when the machine states the
cross-lane unit's queue; without it, each pop follows its push and says why.
    python3 tests/passes/xlu_push_ahead_cases.py   exit 0 iff all hold"""
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
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family import lib_xlu  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.pytorchsim_tpu_family.lower_cross_lane import (  # noqa: E402
    QUEUE_KEY, XLU_PIPELINE)
from pytorchsim_triton_compiler.lowering.passes.lib_ir import walk  # noqa: E402

KERNEL = os.path.join(ROOT, "tests", "cases", "mechanism", "ops", "ops_cumsum_bool_promoted.py")


def _togsim_dir(d, queue):
    """A TOGSim directory like the compiler's, its machine file stating `queue` or not."""
    src = _target.configs_dir()
    top = os.path.dirname(src)
    out = os.path.join(d, "togsim")
    os.makedirs(os.path.join(out, "configs"))
    for x in os.listdir(top):
        if x != "configs":
            os.symlink(os.path.join(top, x), os.path.join(out, x))
    import yaml
    for x in os.listdir(src):
        if x != suite_target.NAME + ".yml":
            os.symlink(os.path.join(src, x), os.path.join(out, "configs", x))
    m = yaml.safe_load(open(os.path.join(src, suite_target.NAME + ".yml")))
    m.setdefault("vcix", {}).pop(QUEUE_KEY, None)
    if queue:
        m["vcix"][QUEUE_KEY] = queue
    yaml.safe_dump(m, open(os.path.join(out, "configs", suite_target.NAME + ".yml"), "w"))
    return out


LOOPS = ("affine.for", "scf.for")


def _lowered(queue):
    """(how many loops hold both an XLU push and a pop, how many pushes there are, why the pass
    did not push ahead or None) -- read off the cross-lane pass's own output."""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, TOGSIM_DIR=_togsim_dir(d, queue))
        w = os.path.join(d, "w")
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchsim-compile"), KERNEL,
                        "--workdir", w, "--to", "lower"], env=env, check=False,
                       capture_output=True)
        out = [f for f in sorted(os.listdir(os.path.join(w, "09-passes")))
               if "lower-cross-lane" in f]
        with ir.Context() as ctx, ir.Location.unknown():
            ctx.allow_unregistered_dialects = True
            m = ir.Module.parse(open(os.path.join(w, "09-passes", out[0])).read())
            why, pushes, holders = None, 0, []
            for o in walk(m.operation):
                if o.name == "func.func" and XLU_PIPELINE in o.attributes:
                    why = ir.StringAttr(o.attributes[XLU_PIPELINE]).value
                push = (o.name == "vcix.iv"
                        and ir.IntegerAttr(o.attributes["opcode"]).value == lib_xlu.XLU_PUSH_OPCODE
                        and lib_xlu.emitter_of(ir.IntegerAttr(o.attributes["imm"]).value & 31)
                        == lib_xlu.XLU_PERMUTE)
                pop = (o.name == "vcix.v.i"
                       and ir.IntegerAttr(o.attributes["opcode"]).value == lib_xlu.XLU_POP_OPCODE)
                if not (push or pop):
                    continue
                pushes += push
                loop = o.parent
                while loop is not None and loop.name not in LOOPS:
                    loop = loop.parent
                if loop is None:
                    continue
                for held in holders:
                    if held[0] == loop:
                        held[1].add("P" if push else "p")
                        break
                else:
                    holders.append((loop, {"P" if push else "p"}))
            both = sum(1 for _, k in holders if k == {"P", "p"})
            return both, pushes, why


def a_stated_queue_pushes_every_round_first():
    both, pushes, why = _lowered(256)
    assert why is None, why
    assert pushes and both == 0, (both, pushes)


def no_stated_queue_pops_each_round_and_says_why():
    both, pushes, why = _lowered(None)
    assert why and QUEUE_KEY in why, why
    assert pushes and both > 0, (both, pushes)


def a_queue_smaller_than_the_rounds_still_pushes_every_round_first():
    """ops_cumsum_bool_promoted's permutes push 64 rows into a queue of 16, as wide as the unit takes a
    cycle. Breaks if the permute serialises once its rows outnumber the queue."""
    both, pushes, why = _lowered(16)
    assert why is None, why
    assert pushes and both == 0, (both, pushes)


CASES = [a_stated_queue_pushes_every_round_first, no_stated_queue_pops_each_round_and_says_why,
         a_queue_smaller_than_the_rounds_still_pushes_every_round_first]


def main():
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"PASS  {case.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
