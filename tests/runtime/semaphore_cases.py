"""togcheck's four semaphore refusals, each on the smallest recorded trace that earns it, and one
trace that earns none: a wrong sum, a re-wait of another count, a wait nothing signaled, an
unwaited load batch.
    python3 tests/runtime/semaphore_cases.py        exit 0 iff every case gets what it names"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import tog_replay  # noqa: E402


def _dma(sem, dims, dir=0, is_async=1):
    return {"op": "dma", "dir": dir, "async": is_async, "sem": sem, "dims": dims, "iv": [0]}


def _bar(sem, expected):
    return {"op": "barrier", "sem": sem, "expected": expected, "iv": [0]}


_ITEM = {"op": "dispatch", "key": 0, "iv": [0]}

CASES = {
    "batch_of_two_waited_once": ([_ITEM, _dma(7, [256, 16]), _dma(7, [1, 16]), _bar(7, 4112),
                                  _bar(7, 4112), _dma(9, [8, 64], dir=1)], None),
    "K3_sum_mismatch": ([_ITEM, _dma(7, [256, 16]), _dma(7, [1, 16]), _bar(7, 16)],
                        "wait expects another element count"),
    "K6_rewait_count": ([_ITEM, _dma(7, [256, 16]), _dma(7, [1, 16]), _bar(7, 4112), _bar(7, 16)],
                        "re-wait expects another element count"),
    "K4_never_signaled": ([_ITEM, _dma(7, [1, 16]), _bar(7, 16), _bar(8, 16)],
                          "wait on a semaphore never signaled"),
    "K5_unwaited_batch": ([_ITEM, _dma(7, [1, 16]), _bar(7, 16), _dma(7, [1, 16])],
                          "unwaited load batch"),
}


def main():
    failed = 0
    for name, (events, want) in CASES.items():
        got = tog_replay.semaphores(events)
        ok = (not got) if want is None else (len(got) == 1 and want in got[0])
        failed += not ok
        print(f"{name:28} {'PASS' if ok else 'FAIL'}  {got if got else 'no refusal'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
