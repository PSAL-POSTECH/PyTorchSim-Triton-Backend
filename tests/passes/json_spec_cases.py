"""A JSON spec, as the Triton backend's pytorchsim_spec option carries it: the graph's on_chip and prefetch reach
the KernelSpec as the file form states them; a field JSON_FIELDS does not name, or a bad index, is refused.
    python3 tests/passes/json_spec_cases.py        exit 0 iff every case holds"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pytorchsim_triton_compiler.contract.kernel_spec import spec_from_json  # noqa: E402

SIGNATURE = {"in_ptr0": "*i8", "in_ptr1": "*i8", "out_ptr0": "*i32", "xnumel": "i32"}
ARGS = [{"role": "in", "shape": [64]}, {"role": "in", "shape": [64]}, {"role": "out", "shape": [64]}]


def _json(**kw):
    return json.dumps(dict(name="probe", signature=SIGNATURE, args=ARGS, grid=[1], **kw))


def on_chip_and_prefetch_reach_the_spec_as_tuples():
    spec = spec_from_json(_json(on_chip=[0, 2], prefetch=[1]))
    assert spec.on_chip == (0, 2) and spec.prefetch == (1,), (spec.on_chip, spec.prefetch)


def a_spec_that_states_neither_keeps_nothing_on_chip_and_copies_nothing_ahead():
    spec = spec_from_json(_json())
    assert spec.on_chip == () and spec.prefetch == (), (spec.on_chip, spec.prefetch)


def a_field_json_fields_does_not_name_is_refused():
    try:
        spec_from_json(_json(on_chip_args=[0]))
    except ValueError as e:
        assert "on_chip_args" in str(e), e
        return
    raise AssertionError("taken: on_chip_args")


def an_index_past_the_arguments_or_a_prefetched_output_is_refused():
    for bad, field in (({"on_chip": [3]}, "on_chip"), ({"prefetch": [2]}, "prefetch"),
                       ({"on_chip": [1], "prefetch": [1]}, "prefetch")):
        try:
            spec_from_json(_json(**bad))
        except ValueError as e:
            assert f"KernelSpec.{field}" in str(e), e
            continue
        raise AssertionError(f"taken: {bad}")


CASES = [on_chip_and_prefetch_reach_the_spec_as_tuples,
         a_spec_that_states_neither_keeps_nothing_on_chip_and_copies_nothing_ahead,
         a_field_json_fields_does_not_name_is_refused,
         an_index_past_the_arguments_or_a_prefetched_output_is_refused]

if __name__ == "__main__":
    failed = 0
    for case in CASES:
        try:
            case()
            print("PASS ", case.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL ", case.__name__, "--", type(e).__name__, e)
    sys.exit(1 if failed else 0)
