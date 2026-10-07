"""Where the wall clock went -- per stage, per pass, per tool."""
import json
import os
import threading
import time
from contextlib import contextmanager

ENV = "TORCHSIM_COMPILE_BREAKDOWN"

FILENAME = "timing.json"


def want_report():
    return os.environ.get(ENV, "0") not in ("0", "", "no", "false")


class Profile:
    """One run's stages, the passes inside them, and the tools they shelled to."""

    def __init__(self):
        self.stages = []
        self.passes = {}
        self.tools = {}
        self.label = None
        self._stack = []
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name):
        """Time one pipeline stage. Records even when the stage raises, which
        is when the number is most worth having."""
        t0 = time.perf_counter()
        self._stack.append(name)
        try:
            yield
        finally:
            self._stack.pop()
            self.stages.append((name, time.perf_counter() - t0))

    def _where(self):
        return self._stack[-1] if self._stack else "?"

    def add_pass(self, name, seconds):
        self.passes.setdefault(self._where(), []).append((name, seconds))

    @contextmanager
    def step(self, name):
        """Time a named piece of a stage that is not one of
        pytorchsim_triton_compiler/lowering/passes/."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_pass(name, time.perf_counter() - t0)

    def add_tool(self, argv0, seconds):
        """Charge `seconds` to the binary at `argv0`, under the current stage."""
        name = os.path.basename(str(argv0))
        with self._lock:
            e = self.tools.setdefault(self._where(), {}).setdefault(name, [0, 0.0])
            e[0] += 1
            e[1] += seconds

    @property
    def total(self):
        return sum(dt for _, dt in self.stages)

    def as_dict(self):
        return {
            "label": self.label,
            "total": self.total,
            "stages": [{"name": n, "seconds": dt} for n, dt in self.stages],
            "passes": {s: [{"name": n, "seconds": dt} for n, dt in v]
                       for s, v in self.passes.items()},
            "tools": {s: {t: {"calls": c, "seconds": dt}
                          for t, (c, dt) in v.items()}
                      for s, v in self.tools.items()},
        }

    def write(self, path):
        """Write the record beside the stage artifacts. Never raises: a run is
        not failed by being unable to say how long it took."""
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.as_dict(), f, indent=2)
            os.replace(tmp, path)
        except OSError:
            pass

    def report(self, width=64):
        return render(self.as_dict(), width=width)


PROF = Profile()


def _bar(frac, width=20):
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def _rows(entries, total, indent, out, width, limit=None, floor=0.0):
    """Print `entries` as `name  seconds  percent`, largest first."""
    entries = sorted(entries, key=lambda e: -e[1])
    shown, rest = entries, []
    if limit is not None and len(entries) > limit:
        shown, rest = entries[:limit], entries[limit:]
    if floor:
        keep = [e for e in shown if e[1] >= floor]
        rest = rest + [e for e in shown if e[1] < floor]
        shown = keep
    for name, dt in shown:
        pct = 100.0 * dt / total if total else 0.0
        out.append(f"{' ' * indent}{name:<{width - indent}} {dt:8.2f}s {pct:5.1f}%")
    if rest:
        dt = sum(e[1] for e in rest)
        pct = 100.0 * dt / total if total else 0.0
        out.append(f"{' ' * indent}{f'({len(rest)} smaller)':<{width - indent}} "
                   f"{dt:8.2f}s {pct:5.1f}%")


def render(rec, width=64):
    """The table a human reads, from the same dict `timing.json` holds."""
    total = rec.get("total") or 0.0
    out = []
    head = "BREAKDOWN"
    if rec.get("label"):
        head += f"  {rec['label']}"
    out.append(f"{head}   total {total:.2f}s")
    out.append("")
    for name, dt in [(s["name"], s["seconds"]) for s in rec.get("stages", [])]:
        pct = 100.0 * dt / total if total else 0.0
        out.append(f"  {name:<12} {dt:8.2f}s {pct:5.1f}%  {_bar(pct / 100.0)}")

    for stage, entries in rec.get("passes", {}).items():
        items = [(e["name"], e["seconds"]) for e in entries]
        if not items:
            continue
        out.append("")
        out.append(f"  inside {stage}")
        _rows(items, total, 4, out, width, floor=0.05)

    tools = rec.get("tools", {})
    if any(tools.values()):
        out.append("")
        out.append("  external tools, summed over calls -- stage 7 runs "
                   "several spikes at once")
        for stage, entries in tools.items():
            for tool, e in sorted(entries.items(), key=lambda kv: -kv[1]["seconds"]):
                out.append(f"    {f'{stage}/{tool}':<{width - 4}} "
                           f"{e['seconds']:8.2f}s  x{e['calls']}")
    return "\n".join(out)


def read(path):
    """A `timing.json` off disk, or None. Never raises -- a caller aggregating
    a hundred workdirs must not die on the one that was killed mid-write."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def merge(records):
    """Sum a list of per-kernel records into one of the same shape."""
    stages, passes, tools = {}, {}, {}
    order = []
    for rec in records:
        if not rec:
            continue
        for s in rec.get("stages", []):
            if s["name"] not in stages:
                order.append(s["name"])
            stages[s["name"]] = stages.get(s["name"], 0.0) + s["seconds"]
        for stage, entries in rec.get("passes", {}).items():
            acc = passes.setdefault(stage, {})
            for e in entries:
                acc[e["name"]] = acc.get(e["name"], 0.0) + e["seconds"]
        for stage, entries in rec.get("tools", {}).items():
            acc = tools.setdefault(stage, {})
            for t, e in entries.items():
                cur = acc.setdefault(t, {"calls": 0, "seconds": 0.0})
                cur["calls"] += e["calls"]
                cur["seconds"] += e["seconds"]
    return {
        "label": None,
        "total": sum(stages.values()),
        "stages": [{"name": n, "seconds": stages[n]} for n in order],
        "passes": {s: [{"name": n, "seconds": dt} for n, dt in v.items()]
                   for s, v in passes.items()},
        "tools": tools,
    }


def sweep(entries):
    """Merge many runs' records into one report, dearest first."""
    import os

    records, per_label = [], []
    for label, wd in entries:
        rec = read(os.path.join(wd, FILENAME))
        if not rec:
            continue
        records.append(rec)
        per_label.append((label, rec.get("total") or 0.0))

    if not records:
        return None

    merged = merge(records)
    merged["label"] = f"{len(records)} kernel(s)"
    out = [render(merged), "", "  dearest kernels"]
    for label, dt in sorted(per_label, key=lambda k: -k[1])[:10]:
        pct = 100.0 * dt / merged["total"] if merged["total"] else 0.0
        out.append(f"    {label:<60} {dt:8.2f}s {pct:5.1f}%")
    return "\n".join(out)
