#!/usr/bin/env python3
"""The layer suite's CI cap: --ci times its layers and its MoE ops (CI_OPS) up to CI_TOKENS tokens, and none of the
ops Qwen3-8B also has. python3 validation/layer/test_cap.py (in the compiler image)."""
import csv
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("layer_simulate_shapes", os.path.join(HERE, "simulate_shapes.py"))
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class Cap(unittest.TestCase):
    def test_layers_and_moe_ops_alone(self):
        with open(os.path.join(HERE, "tpu_v6e_qwen3moe_layer_torchxla.csv")) as fh:
            rows = [r for r in csv.DictReader(fh) if r["dtype_in"] == "bf16"]
        ci = [r for r in rows if suite.under_cap({k: r[k] if k in ("section", "name") else int(r[k])
                                                  for k in suite.KEYS})]
        self.assertEqual({r["name"] for r in ci if r["section"] == "op"}, set(suite.CI_OPS))
        self.assertEqual({r["name"] for r in ci if r["section"] == "layer"}, {"prefill", "decode"})
        self.assertFalse(suite.under_cap({"section": "op", "name": "rmsnorm", "T": 128, "S": 0}))
        self.assertFalse(suite.under_cap({"section": "op", "name": "combine", "T": 8192, "S": 0}))


if __name__ == "__main__":
    unittest.main()
