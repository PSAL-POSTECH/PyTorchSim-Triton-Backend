#!/usr/bin/env python3
"""results.json + shapes.csv + the measured v6e Qwen3-8B layer rows -> report.html, summary.md and results.csv, as
validation/layer reports Qwen3-30B-A3B (its emit_report, with this model): the decoder layer as one program, each
op alone, and each layer against the sum of its ops at its T."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from suite_module import load  # noqa: E402


#: How often the layer runs each measured op (PyTorchSim's scripts/ci/bundle.py `_dense_layer`, as the archive's
#: harness/qwen3_8b_layer_v1.py writes it): rmsnorm before attention and before the MLP, both residuals, the MLP's
#: gate and up as one weight; each layer kind's attention is its op at the layer's T and S.
MODEL = {
    "name": "Qwen3-8B", "csv": os.path.join(HERE, "tpu_v6e_qwen3_8b_layer_torchxla.csv"),
    "ops": {"rmsnorm": 2, "q_proj": 1, "k_proj": 1, "v_proj": 1, "q_norm": 1, "k_norm": 1, "rope_q": 1, "rope_k": 1,
            "o_proj": 1, "residual": 2, "gate_up": 1, "silu_mul": 1, "down": 1},
    "attention": {"prefill": "attn_prefill", "decode": "attn_decode"},
    "uncovered": None,
    "warn": "layers and attention ops past S 1024 had no CPU check, and some small ops' p10–p90 is wide (its reason "
            "gives it)"}

report = load("layer", "emit_report")

if __name__ == "__main__":
    sys.exit(report.main(model=MODEL))
