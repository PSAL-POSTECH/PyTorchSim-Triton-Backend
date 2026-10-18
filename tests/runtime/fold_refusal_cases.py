"""A kernel with fewer rows than lanes that torchsim-fold-rows-onto-lanes leaves one row a lane is not
left silently: the pass names the op that stopped it on the module (torchsim.rows_not_folded) and the
triton-compile stage logs it. A kernel the pass folds carries no such name.
    python3 tests/runtime/fold_refusal_cases.py        exit 0 iff every case gets what it names"""
import json
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

ATTR = "torchsim.rows_not_folded"


def _triton_compile(kernel, workdir):
    """The op the fold named on 07-regularized's module (None when it named none), and whether the
    stage log reports it."""
    subprocess.run([sys.executable, os.path.join(ROOT, "pytorchsim-triton-compiler"),
                    os.path.join(ROOT, kernel), "--workdir", workdir, "--to", "transform"],
                   capture_output=True, text=True)
    toolchain.ensure_bindings()
    from mlir.ir import Context, Module, StringAttr
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, open(os.path.join(workdir, "07-regularized.mlir")) as f:
        attrs = Module.parse(f.read()).operation.attributes
        named = StringAttr(attrs[ATTR]).value if ATTR in attrs else None
    log = open(os.path.join(workdir, "stage.log")).read()
    return named, "torchsim-fold-rows-onto-lanes stopped at" in log


def _refused_kernel_names_its_op():
    # 16 rows of 65536 and a transposed read of x: the transpose is a use the fold does not take.
    with tempfile.TemporaryDirectory() as d:
        named, logged = _triton_compile("tests/cases/mechanism/tile/tile_spad_over_budget.py", d)
    return named == "tt.trans" and logged, {"named": named, "logged": logged}


def _folded_kernel_names_none():
    with tempfile.TemporaryDirectory() as d:
        named, logged = _triton_compile(
            "tests/cases/mechanism/reduce/reduce_one_row_across_lanes.py", d)
    return named is None and not logged, {"named": named, "logged": logged}


CASES = {"refused_kernel_names_its_op": _refused_kernel_names_its_op,
         "folded_kernel_names_none": _folded_kernel_names_none}


def main():
    failed = 0
    for name, case in CASES.items():
        try:
            ok, got = case()
        except Exception as e:  # noqa: BLE001
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:36} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:300]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
