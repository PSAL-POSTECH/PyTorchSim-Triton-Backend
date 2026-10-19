# PyTorchSim-Triton-Backend: notes for Claude

The compiler (PTB) lowers Triton kernels to RISC-V + the accelerator's custom instructions, and TOGSim times them.
`validation/` times the kernels PyTorchSim builds and selects (`validation/pytorchsim`, proposed by PyTorchSim's CI)
against TPU v6e measurements (`validation/*/tpu_v6e_*.csv`) and against main's last results.

## Validation error loop

**Objective:** for every target group, the geomean of TOGSim time ÷ measured time within 0.85..1.15. The groups,
the band and the exclusions are `validation/error_targets.json` (read by `validation/error_targets.py`); an excluded
group is shown and not judged, and a workload no group takes is listed as ungrouped. Changing that file is the
maintainer's call, not the loop's.
The share within ±20% is shown beside it: a geomean can hide a fast half and a slow half, so read both.

The tracker issue "Validation error tracker" holds main's latest table and, as comments, its history. When a group is
off target and no pull request labeled `error-loop` is open, main's run asks @claude for the next iteration.

One iteration:
1. **Pick** the off-target group with the most workloads outside ±20%. One group per iteration.
2. **Explain before changing anything.** Set the simulated time's parts (TOGSim units' busy cycles, DMA, gem5 tile
   cycles, launches: the report's results.json and logs/ in the run's `<suite>-validation-<sha>` artifact) beside the
   measured ones (the table's columns: kernel_us, compute_us, dma, lead_in_ns, ...). Name which part disagrees and
   why, citing a log line, a number from an artifact, or `file:line`. If the artifacts cannot answer it, say what
   measurement would, and stop there.
3. **Place the cause** in one layer, and act only where it is ours:
   - measurement method in `validation/` (what the TPU number includes that the simulation does not) — fix here;
   - PTB lowering (the same program scheduled or moved differently than v6e's LLO does) — fix here;
   - PyTorchSim's kernel (a different program than the TPU ran: blocks, program count) — describe it in the issue
     for PyTorchSim; do not change PyTorchSim from here;
   - TOGSim or the accelerator model's timing — never change it here; open an issue with the evidence.
4. **Fix** on one branch, `claude/error-loop-<group>`, and link the pull request to open; a person opens it as a draft
   labeled `error-loop`. Its description states the cause, the evidence and the group's geomean before. Its CI
   posts the groups' table on the pull request: report the change, including any other group it moved.
5. Never merge. A person reviews and merges (and deletes the branch); main's next run measures again. While an
   `error-loop` pull request is open or a `claude/error-loop-*` branch is left, no new iteration starts.

Rules:
- No fitted constants: never tune a latency, rate or factor to move a ratio. A timing value changes only with a
  measurement behind it (the TPU archive, LLO bundles), cited.
- Report only what is shown. Label anything else as a hypothesis, and do not build on it.
- Keep a change to its cause; delete what it makes obsolete rather than adding a layer.
- Never parse kernel source text; read structured data (specs, manifests, IR through MLIR, JSON).
- One iteration in flight at a time.
