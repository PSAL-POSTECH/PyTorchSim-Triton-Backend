"""The loop every case script under tests/passes, tests/runtime and tests/tools ends in: run each case, print
one line for it, and exit nonzero when any failed. tests/run.py reads the FAIL lines."""
import contextlib
import json

WIDTH = 300


def _mlir(unregistered):
    from mlir.ir import Context, Location
    stack = contextlib.ExitStack()
    ctx = stack.enter_context(Context())
    ctx.allow_unregistered_dialects = unregistered
    stack.enter_context(Location.unknown())
    return stack


def run(cases, mlir=False, unregistered=False):
    """Cases that fail by raising, each inside a fresh MLIR context when `mlir` or `unregistered` says so."""
    failed = 0
    for case in cases:
        try:
            with _mlir(unregistered) if (mlir or unregistered) else contextlib.nullcontext():
                case()
            print(f"PASS  {case.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:WIDTH]}")
    return 1 if failed else 0


def run_checks(cases):
    """{name: case} where a case returns (ok, what it got)."""
    width = max(map(len, cases), default=0)
    failed = 0
    for name, case in cases.items():
        try:
            ok, got = case()
        except Exception as e:
            ok, got = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{name:{width}} {'PASS' if ok else 'FAIL'}  {json.dumps(got, default=str)[:WIDTH]}")
    return 1 if failed else 0


def run_wants(cases):
    """{name: (case, want)}: a string want is looked for in a string result, anything else compared."""
    width = max(map(len, cases), default=0)
    failed = 0
    for name, (case, want) in cases.items():
        try:
            got = case()
        except Exception as e:
            got = f"{type(e).__name__}: {e}"
        ok = want in got if isinstance(want, str) and isinstance(got, str) else got == want
        failed += not ok
        print(f"{name:{width}} {'PASS' if ok else 'FAIL'}  got {str(got)[:WIDTH]}  want {str(want)[:WIDTH]}")
    return 1 if failed else 0
