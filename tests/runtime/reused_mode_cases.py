"""A tool output the compile writes again from its cache keeps the mode the tool gave it: a linked ELF stays
executable, which QEMU linux-user checks before it loads a program (pk loads it either way).
    python3 tests/runtime/reused_mode_cases.py        exit 0 iff every case holds"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pytorchsim_triton_compiler.driver import stage  # noqa: E402

TOOL = """#!/bin/sh
while [ "$1" != -o ]; do shift; done
echo run >> "$(dirname "$0")/runs"
printf 'ELF' > "$2"
chmod 755 "$2"
"""


class _Ctx:
    """What run_tool reads of a compile: a log and the cache of tool outputs."""

    def __init__(self):
        self.tool_reuse = {}

    def log(self, _):
        pass


def a_reused_output_keeps_its_execute_bits(d):
    """Breaks if the second call runs the tool again instead of reusing it, or if the reused output is written
    back without the mode the tool left it (0755 read back as 0644, the ELF QEMU refused with ENOEXEC)."""
    tool = os.path.join(d, "riscv64-unknown-elf-gcc")
    with open(tool, "w") as f:
        f.write(TOOL)
    os.chmod(tool, 0o755)
    out, ctx = os.path.join(d, "k.elf"), _Ctx()
    stage.run_tool(ctx, "binary", [tool, "-o", out])
    os.remove(out)
    stage.run_tool(ctx, "binary", [tool, "-o", out])
    runs = open(os.path.join(d, "runs")).read().split()
    assert len(runs) == 1, f"the tool ran {len(runs)} times"
    assert os.stat(out).st_mode & 0o777 == 0o755, oct(os.stat(out).st_mode & 0o777)


CASES = [a_reused_output_keeps_its_execute_bits]


def main():
    bad = 0
    for case in CASES:
        with tempfile.TemporaryDirectory() as d:
            try:
                case(d)
                print(f"PASS  {case.__name__}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"FAIL  {case.__name__}: {type(e).__name__}: {str(e)[:400]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
