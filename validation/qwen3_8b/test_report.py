#!/usr/bin/env python3
"""The Qwen3-8B suite through validation/layer's runner and report: a layer is set against all its ops at its T with
its attention op at its T and S, each attention op has a column of its own S, and --where section=op leaves the
layers out as not picked (not run, not failed). python3 validation/qwen3_8b/test_report.py (in the compiler image)."""
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from suite_module import load  # noqa: E402


suite = load("qwen3_8b", "simulate_shapes")
wrapper = load("qwen3_8b", "emit_report")
report, MODEL = wrapper.report, wrapper.MODEL
REAL = report.measured(MODEL["csv"])
#: (name, section, op, T, S) of the bundle: the decode layer at S 1024, and its attention op at two S.
WORKLOADS = [("qwen3_8b_layer_decode_1x1024_bf16", "layer", "decode", 1, 1024),
             ("qwen3_8b_op_attn_decode_1x1024_bf16", "op", "attn_decode", 1, 1024),
             ("qwen3_8b_op_attn_decode_1x2048_bf16", "op", "attn_decode", 1, 2048),
             ("qwen3_8b_op_rmsnorm_1x0_bf16", "op", "rmsnorm", 1, 0)]


def write_bundle(root):
    with open(os.path.join(root, "bundle.json"), "w") as fh:
        json.dump({"torch": importlib.metadata.version("torch")}, fh)
    for name, section, op, T, S in WORKLOADS:
        os.makedirs(os.path.join(root, name, "00_k"))
        with open(os.path.join(root, name, "manifest.json"), "w") as fh:
            json.dump({"workload": name, "suite": "qwen3_8b", "section": section, "name": op, "T": T, "S": S,
                       "launches": [{"kernel": "k", "dir": "00_k"}], "externs": []}, fh)


class Report(unittest.TestCase):
    def rows(self, conditions=None):
        """The report's rows of the bundle, every workload --where keeps timed at 1 µs."""
        with tempfile.TemporaryDirectory() as tmp:
            bundle, gen = os.path.join(tmp, "bundle"), os.path.join(tmp, "generated")
            os.makedirs(bundle)
            write_bundle(bundle)
            shapes = suite.sim.load_bundle(bundle, "qwen3_8b", suite.KEYS, gen, suite.token_pairs, suite.under_cap,
                                           suite.sim.where(conditions, suite.KEYS))
            results = {"shapes": [{"kernel": s["kernel"], "rc": 0, "us": 1.0} for s in shapes if s["picked"]]}
            return report.join_rows(os.path.join(gen, "shapes.csv"), REAL, results)

    def test_a_layer_is_set_against_every_op_and_its_attention(self):
        d = next(d for d in report.decompose(self.rows(), REAL, MODEL) if (d["name"], d["S"]) == ("decode", 1024))
        want = sum(int(REAL[("op", op, 1, 0)]["latency_ns"]) * n for op, n in MODEL["ops"].items())
        want += int(REAL[("op", "attn_decode", 1, 1024)]["latency_ns"])
        self.assertEqual(d["ops"], len(MODEL["ops"]) + 1)
        self.assertAlmostEqual(d["ops_real_us"], want / 1000)

    def test_each_attention_op_has_its_own_s(self):
        rows = self.rows()
        summary = report.emit_summary(rows, REAL, report.count_chips(rows, []), {"commit": "x" * 12, "ran_on": "t"},
                                      model=MODEL)
        line = next(l for l in summary.splitlines() if l.startswith("| attn_decode |"))
        self.assertEqual(sum(c.strip() not in ("–", "") for c in line.split("|")[2:-1]), 2, line)

    def test_where_leaves_the_layers_out_as_not_picked(self):
        rows = {r["kernel"]: r for r in self.rows(["section=op"])}
        layer = rows[WORKLOADS[0][0]]
        self.assertTrue(layer["skipped"])
        self.assertEqual(layer["rc"], "not picked")
        chips = {label: value for value, label, _ in report.count_chips(list(rows.values()), [])}
        self.assertEqual(chips["not picked, not run"], "1")
        self.assertNotIn("simulation failed", chips)

    def test_where_refuses_a_column_the_suite_has_not(self):
        with self.assertRaises(SystemExit):
            suite.sim.where(["kind=op"], suite.KEYS)


if __name__ == "__main__":
    unittest.main()
