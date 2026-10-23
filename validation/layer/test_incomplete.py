#!/usr/bin/env python3
"""A layer workload whose manifest lists externs (calls Inductor left to aten) is incomplete: simulate_shapes carries
them to externs.json, emit_report keeps it out of every chip, geomean and the decomposition and says why, and
error_targets leaves it out of its group. python3 validation/layer/test_incomplete.py (in the compiler image)."""
import importlib.metadata
import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GMM, NORM = "layer_op_gmm_down_128x0_bf16", "layer_op_rmsnorm_128x0_bf16"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shapes_mod = load("layer_simulate_shapes", os.path.join(HERE, "simulate_shapes.py"))
report = load("layer_emit_report", os.path.join(HERE, "emit_report.py"))
targets = load("error_targets", os.path.join(os.path.dirname(HERE), "error_targets.py"))


def write_bundle(root):
    """A bundle of two ops at T 128: gmm_down, whose grouped matmul is an extern, and rmsnorm, all kernels."""
    with open(os.path.join(root, "bundle.json"), "w") as fh:
        json.dump({"torch": importlib.metadata.version("torch")}, fh)
    for name, op, kernel, externs in [(GMM, "gmm_down", "k_cumsum", [{"call": "aten._grouped_mm.default", "at": 1}]),
                                      (NORM, "rmsnorm", "k_norm", [])]:
        os.makedirs(os.path.join(root, name, "00_" + kernel))
        with open(os.path.join(root, name, "manifest.json"), "w") as fh:
            json.dump({"workload": name, "suite": "layer", "section": "op", "name": op, "T": 128, "S": 0,
                       "launches": [{"kernel": kernel, "dir": "00_" + kernel}], "externs": externs}, fh)


class Incomplete(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        bundle, self.gen = os.path.join(self.tmp.name, "bundle"), os.path.join(self.tmp.name, "generated")
        os.makedirs(bundle)
        write_bundle(bundle)
        shapes_mod.sim.load_bundle(bundle, "layer", shapes_mod.KEYS, self.gen, shapes_mod.token_pairs,
                                   shapes_mod.under_cap)
        self.real = report.measured(os.path.join(HERE, "tpu_v6e_qwen3moe_layer_torchxla.csv"))
        us = {GMM: 2.0, NORM: 1.5}
        results = {"shapes": [{"kernel": k, "rc": 0, "us": v, "total_cycles": int(v * 1000)} for k, v in us.items()]}
        with open(os.path.join(self.gen, "externs.json")) as fh:
            self.externs = json.load(fh)
        self.rows = report.join_rows(os.path.join(self.gen, "shapes.csv"), self.real, results, self.externs)
        self.by = {r["kernel"]: r for r in self.rows}

    def test_externs_reach_the_report(self):
        self.assertEqual(self.externs, {GMM: ["aten._grouped_mm.default"]})
        self.assertIn("aten._grouped_mm.default", self.by[GMM]["incomplete"])
        self.assertEqual(self.by[NORM]["incomplete"], "")

    def test_no_chip_or_geomean_counts_it(self):
        chips = {label: value for value, label, _ in report.count_chips(self.rows, [])}
        self.assertEqual(chips["ops ratio geomean"], f"{self.by[NORM]['ratio']:.3f}")
        self.assertEqual(chips["complete, within ±20%"].split(" / ")[1], "1")
        self.assertEqual(chips["incomplete (calls left to aten), not judged"], "1 / 2")
        summary = report.emit_summary(self.rows, self.real, report.count_chips(self.rows, []), {"commit": "x" * 12,
                                      "ran_on": "test"})
        self.assertIn(f"`{GMM}`: " + self.by[GMM]["incomplete"], summary)

    def test_decomposition_counts_it_untimed(self):
        dec = report.decompose(self.rows, self.real)
        self.assertTrue(all(d["ops_timed"] <= 1 and d["ops_sim_us"] is None for d in dec if d["T"] == 128))

    def test_error_targets_leave_it_out(self):
        report.emit_csv(os.path.join(self.gen, "results.csv"), self.rows)
        with open(os.path.join(HERE, "..", "error_targets.json")) as fh:
            config = json.load(fh)
        ratios, part, _ = targets.assign(config, {"layer": targets.rows_of(self.gen)})
        group = next(g["name"] for g in config["groups"] if g["suite"] == "layer" and GMM in ratios[g["name"]])
        s = targets.summarize(ratios[group], config["target"], config["spread"], False, part[group])
        self.assertIsNone(s["geomean"])
        self.assertEqual(list(s["incomplete"]), [GMM])
        self.assertIn(GMM, targets.markdown(config, {g["name"]: s if g["name"] == group else targets.summarize(
            ratios[g["name"]], config["target"], config["spread"], False, part[g["name"]]) for g in config["groups"]},
            {}))


if __name__ == "__main__":
    unittest.main()
