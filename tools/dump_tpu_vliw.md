# dump_tpu_vliw

`tools/dump_tpu_vliw.py` dumps what Google's TPU compiler (libtpu) generates for one workload, on a
host with no TPU. Use it to put the real TPU's code next to what this compiler emits for the same
shape.

## Set up (once)

libtpu is about 150 MB and only this tool needs it, so it is not part of the default toolchain or
the image. JAX 0.11.2 needs Python 3.12 or newer; the toolchain's own Python is 3.11.

```bash
JAX_TPU_PYTHON=/path/to/python3.13 setup/toolchain.sh jaxtpu
```

This makes `$TORCHSIM_PREFIX/jax-tpu-venv` from `setup/requirements-jax-tpu.txt` (jax 0.11.2,
libtpu 0.0.48, exact pins).

## Run

```bash
$TORCHSIM_PREFIX/jax-tpu-venv/bin/python -I tools/dump_tpu_vliw.py OUT matmul M K N
```

| Option | Default | Meaning |
|---|---|---|
| `--dtype bf16\|f32` | `bf16` | element type of both operands |
| `--topology` | `v6e=1x1` | TPU generation; `v5e=1x1` and `v5p=1x1` also compile |

`OUT` must not exist or must be empty. Run one dump at a time: libtpu holds `/tmp/libtpu_lockfile`,
and a second run fails on it. Warnings about `TPU_ACCELERATOR_TYPE` and `TPU_WORKER_HOSTNAMES` are
harmless.

Example: the M=1 matrix-vector product of 2048 x 2048.

```bash
$TORCHSIM_PREFIX/jax-tpu-venv/bin/python -I tools/dump_tpu_vliw.py /tmp/gemv2048 matmul 1 2048 2048
```

## What it writes

| Path | Content |
|---|---|
| `OUT/manifest.json` | workload, shapes, dtype, topology, package versions, dump flags. Written last: if it is there, the dump finished. |
| `OUT/hlo/` | XLA's own HLO dumps (`--xla_dump_to`) |
| `OUT/llo/` | libtpu's low-level program, several hundred files |

The files under `OUT/llo/` are named `<id>-<op>-<NN>-<kind>.txt`. `<op>` is an HLO instruction of
the program (for an M=1 matmul: `reduce.3`, `multiply_reduce_fusion`, `reshape.1`), or `TLP` for the
whole program. The ones to open first:

| File | Content |
|---|---|
| `*-TLP-hlo.txt` | the optimized HLO: which ops the program is, with their layouts |
| `*-<op>-NN-final_bundles.txt` | the final VLIW bundles of that op, one bundle per line: `{ a ;; b ;; ... }`. A DMA line carries its window (`window_bounds`, `iteration_bounds`). |
| `*-<op>-NN-schedule-analysis_final_bundles.txt` | how many bundles the op has |
| `*-<op>-NN-static_gap_analysis_final_bundles.txt` | distances between loads and their uses |
| `*-TLP-NN-hlo-static-bundle-profile.txt` | per HLO instruction, how many of each slot kind its bundles use (vector ALU, load, store, XLU, MXU) |

The other files are the same op after each earlier pass.

## Limits

- **A static schedule.** It says which instructions share a bundle, not how many cycles a bundle
  takes or how long a DMA is waited for.
- **Not the measured program, exactly.** The real-TPU measurements used torch_xla with libtpu
  0.0.21; this goes through JAX with libtpu 0.0.48. The op names match (`reduce.3`,
  `multiply_reduce_fusion`, `reshape.1`); compare `*-TLP-hlo.txt` with the measured run's HLO
  before relying on more.
- **The dumps are text, and nothing here parses them.** The tool only writes them.
- An unknown topology string ends in JAX's own error, not a one-line refusal.
