"""Stage 7 — compare Spike's output against the torch reference.

Each kernel file declares `reference(inputs) -> dict[name, tensor]` using the
ordinary torch op it is supposed to implement (torch.matmul, torch.bmm, x + y,
...). This stage runs that on the CPU, reads back the .raw files Spike wrote,
and reports a per-tensor comparison.

Failure output is a report, not just False: max abs/rel error, the index where
it happens, and both values -- that is what you need when a lowering pass is
subtly wrong rather than catastrophically wrong.
"""
import numpy as np

from pytorchsim_triton_compiler.contract.kernel_object import NP_DTYPE as _NAMES

NP_DTYPE = {k: getattr(np, v) for k, v in _NAMES.items()}


def read_raw(path, arg):
    data = np.fromfile(path, dtype=NP_DTYPE[arg.dtype], count=arg.numel)
    if data.size != arg.numel:
        raise ValueError(
            f"{path}: read {data.size} elements, expected {arg.numel} "
            f"(file truncated -- did the kernel actually write its output?)")
    return data.reshape(arg.shape)


def compare(spec, actual_paths, inputs, torch):
    """Returns (all_ok, list_of_report_lines)."""
    expected = spec.reference(inputs)
    lines = []
    all_ok = True

    for arg in spec.outputs():
        got = read_raw(actual_paths[arg.name], arg)
        w = expected[arg.name].detach().cpu()
        if arg.dtype == "bfloat16":
            got = (torch.from_numpy(np.ascontiguousarray(got))
                   .view(torch.bfloat16).float().numpy())
            w = w.float()
        want = w.numpy().reshape(arg.shape)

        got_t = torch.from_numpy(np.ascontiguousarray(got))
        want_t = torch.from_numpy(np.ascontiguousarray(want))

        ok = torch.allclose(got_t.float(), want_t.float(),
                            rtol=spec.rtol, atol=spec.atol, equal_nan=True)
        all_ok &= ok

        g64, w64 = got.astype(np.float64), want.astype(np.float64)
        same = (g64 == w64) | (np.isnan(g64) & np.isnan(w64))
        diff = np.where(same, 0.0, np.abs(g64 - w64))
        denom = np.maximum(np.abs(want.astype(np.float64)), 1e-12)
        rel = diff / denom
        idx = int(np.argmax(diff)) if diff.size else 0
        flat_got, flat_want = got.reshape(-1), want.reshape(-1)

        lines.append(f"  {arg.name:<10} {'PASS' if ok else 'FAIL'}  "
                     f"shape={tuple(arg.shape)} dtype={arg.dtype}")
        lines.append(f"    max_abs_err = {diff.max() if diff.size else 0:.6g}   "
                     f"max_rel_err = {rel.max() if rel.size else 0:.6g}   "
                     f"(rtol={spec.rtol}, atol={spec.atol})")
        if not ok and diff.size:
            lines.append(f"    worst at flat index {idx}: "
                         f"got={flat_got[idx]!r}  want={flat_want[idx]!r}")
            nbad = int((diff > (spec.atol + spec.rtol * np.abs(want.astype(np.float64)))).sum())
            lines.append(f"    {nbad}/{diff.size} elements outside tolerance")

    return all_ok, lines
