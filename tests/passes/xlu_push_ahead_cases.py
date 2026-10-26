"""lower_xlu_moves pushes every round of a permute before its first pop when the machine states the
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
os.environ.setdefault("PYTORCHSIM_COMPILE_TARGET", suite_target.NAME)
from pytorchsim_triton_compiler import toolchain  # noqa: E402
from pytorchsim_triton_compiler.contract import target as _target  # noqa: E402

toolchain.ensure_bindings()
from mlir import ir  # noqa: E402
from pytorchsim_triton_compiler.dialects.pytorchsim_tpu import XluPopOp, XluPushOp  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.tpu_kind.lib_xlu_move import QUEUE_KEY  # noqa: E402
from pytorchsim_triton_compiler.lowering.ext.tpu_kind.lower_xlu_moves import XLU_PIPELINE  # noqa: E402
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


def _block_of(o):
    """The loop or function body `o` sits in, as the op that holds it."""
    p = o.parent
    while p is not None and p.name not in ("affine.for", "scf.for", "func.func"):
        p = p.parent
    return p


def _lowered(queue):
    """(the most permute rows a block has pushed and not yet popped, in program order, how many
    pushes there are, why the pass did not push ahead or None) -- read off the cross-lane pass's
    own output. A push spans a register group, so a move's pushes and pops may share a block with
    no loop between them: what says it pushed ahead is that rows wait in the queue, more than one."""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, TOGSIM_DIR=_togsim_dir(d, queue))
        w = os.path.join(d, "w")
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "pytorchsim-compile"), KERNEL,
                        "--workdir", w, "--to", "lower"], env=env, check=False,
                       capture_output=True)
        out = [f for f in sorted(os.listdir(os.path.join(w, "09-passes")))
               if "convert-pytorchsim-to-target-kind" in f]
        with ir.Context() as ctx, ir.Location.unknown():
            ctx.allow_unregistered_dialects = True
            m = ir.Module.parse(open(os.path.join(w, "09-passes", out[0])).read())
            why, pushes, waiting, deepest = None, 0, {}, 0
            for o in walk(m.operation):
                if o.name == "func.func" and XLU_PIPELINE in o.attributes:
                    why = ir.StringAttr(o.attributes[XLU_PIPELINE]).value
                push = (o.name == XluPushOp.OPERATION_NAME
                        and ir.StringAttr(o.attributes["mode"]).value == "permute")
                pop = o.name == XluPopOp.OPERATION_NAME
                block = _block_of(o)
                if push:
                    pushes += 1
                    waiting[block] = waiting.get(block, 0) + 1
                    deepest = max(deepest, waiting[block])
                elif pop:
                    waiting[block] = max(0, waiting.get(block, 0) - 1)
            return deepest, pushes, why


def a_stated_queue_pushes_every_round_first():
    deepest, pushes, why = _lowered(256)
    assert why is None, why
    assert pushes and deepest > 1, (deepest, pushes)


def no_stated_queue_pops_each_round_and_says_why():
    deepest, pushes, why = _lowered(None)
    assert why and QUEUE_KEY in why, why
    assert pushes and deepest == 1, (deepest, pushes)


def a_queue_smaller_than_the_rounds_still_pushes_every_round_first():
    """ops_cumsum_bool_promoted's permutes push 64 rows into a queue of 16, as wide as the unit takes a
    cycle. Breaks if the permute serialises once its rows outnumber the queue."""
    deepest, pushes, why = _lowered(16)
    assert why is None, why
    assert pushes and deepest > 1, (deepest, pushes)


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
