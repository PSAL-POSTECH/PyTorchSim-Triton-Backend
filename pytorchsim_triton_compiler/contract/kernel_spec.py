"""What a kernel file must declare."""
import math

from .kernel_object import ITEMSIZE
from dataclasses import dataclass, field
from typing import Callable

IN = "in"
OUT = "out"
INOUT = "inout"

OFFSET_INDEX_BITS = 32
"""The premise, not verified: every index a kernel computes at this width fits in it, as the frontend
indexes at i32 only when every size does. The compiler marks such index arithmetic nsw."""

def is_read(arg):
    return arg.role in (IN, INOUT)


def is_written(arg):
    return arg.role in (OUT, INOUT)


@dataclass
class Arg:
    """One kernel argument that is backed by memory (a tensor)."""
    name: str
    role: str
    dtype: str
    shape: tuple

    @property
    def numel(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.numel * ITEMSIZE[self.dtype]


@dataclass
class KernelSpec:
    name: str
    kernel: Callable
    signature: dict
    constexprs: dict
    args: list
    grid: tuple
    reference: Callable
    make_inputs: Callable
    rtol: float = 1e-4
    atol: float = 1e-4
    expect: str = "pass"
    expect_why: str = ""
    expect_ir: dict = field(default_factory=dict)
    expect_ir_on: dict = field(default_factory=dict)
    expect_stage: str = ""
    expect_stop: object = None
    parallel_axes: tuple = ()
    extra: dict = field(default_factory=dict)
    grid_rule: bool = True
    """The kernel follows the grid rule (an axis covers its numel in tile steps), so the compile reads each
    axis's program-id bound from the IR; False keeps `grid` as the bound the compile assumes."""
    ttir: str = ""
    """A path to the Triton IR the frontend starts from, when the kernel was already traced by a Triton
    that called this backend (triton_extension); "" runs `kernel` through the Triton frontend."""
    graph_prefetch: object = None
    """The arguments (spec.args indices) this kernel's copy node holds, its graph prefetching them: () none,
    the prefetch being elsewhere in the graph; None, the compiler's own choice in the kernel."""
    on_chip: tuple = ()
    """The arguments (spec.args indices) the graph keeps in the scratchpad across kernels: an input a kernel
    before this one left there, an output left for a later one. The TOG moves none of them; the binary does."""
    prefetch: tuple = ()
    """The inputs (spec.args indices) the graph copies into the scratchpad ahead of this kernel, beside the
    kernels before it: each a copy node, which the TOG reads instead of moving it. The binary still moves it."""

    def __post_init__(self):
        if not self.parallel_axes:
            self.parallel_axes = ("x", "y", "z")[:max(1, len(self.grid))]
        bad = [a for a in self.parallel_axes if a not in ("x", "y", "z")]
        if bad or len(set(self.parallel_axes)) != len(self.parallel_axes):
            raise ValueError(
                f"KernelSpec.parallel_axes is a set of distinct 'x'/'y'/'z', "
                f"outermost first; got {self.parallel_axes!r}")
        if self.graph_prefetch is not None:
            got = tuple(self.graph_prefetch)
            ok = all(isinstance(a, int) and 0 <= a < len(self.args) for a in got)
            if not ok or len(set(got)) != len(got):
                raise ValueError(f"KernelSpec.graph_prefetch is None or distinct indices into args, not "
                                 f"{self.graph_prefetch!r}: {self.name}")
            self.graph_prefetch = got
        got = tuple(self.on_chip)
        if not all(isinstance(a, int) and 0 <= a < len(self.args) for a in got) or len(set(got)) != len(got):
            raise ValueError(f"KernelSpec.on_chip is distinct indices into args, not {self.on_chip!r}: {self.name}")
        self.on_chip = got
        got = tuple(self.prefetch)
        if (not all(isinstance(a, int) and 0 <= a < len(self.args) and self.args[a].role == IN for a in got)
                or len(set(got)) != len(got) or set(got) & (set(self.on_chip) | set(self.graph_prefetch or ()))):
            raise ValueError(f"KernelSpec.prefetch is distinct indices of inputs, none on_chip, not "
                             f"{self.prefetch!r}: {self.name}")
        self.prefetch = got
        if self.expect not in ("pass", "refuse", "blocked", "open"):
            raise ValueError(
                "KernelSpec.expect is 'pass', 'refuse', 'blocked' or 'open', "
                "not %r" % self.expect)
        if self.expect_stage not in ("", "togcheck", "timing"):
            raise ValueError(
                "KernelSpec.expect_stage names the stage whose run is what this file pins: "
                "'togcheck' or 'timing', not %r" % self.expect_stage)
        if self.expect != "pass" and not self.expect_why:
            raise ValueError(
                "a kernel that is not expected to pass has to say why "
                "(expect_why): %s" % self.name)
        from .stop import Stop
        if (self.expect == "pass") != (self.expect_stop is None):
            raise ValueError(
                "a refuse, blocked or open kernel says where it stops and why (expect_stop, a "
                "contract.stop.Stop), and a passing one does not: %s" % self.name)
        if self.expect_stop is not None and not isinstance(self.expect_stop, Stop):
            raise ValueError("KernelSpec.expect_stop is a contract.stop.Stop: %s" % self.name)
        if not all(isinstance(m, str) and isinstance(files, dict)
                   and all(isinstance(c, dict) for c in files.values())
                   for m, files in self.expect_ir_on.items()):
            raise ValueError("KernelSpec.expect_ir_on maps a machine's name to expect_ir's "
                             "{artifact: {key: count}}: %s" % self.name)

    def expect_ir_for(self, machine_name):
        """expect_ir with expect_ir_on[machine_name]'s counts laid over it, key by key."""
        out = {f: dict(c) for f, c in (self.expect_ir or {}).items()}
        for f, counts in self.expect_ir_on.get(machine_name, {}).items():
            out.setdefault(f, {}).update(counts)
        return out

    def inputs(self):
        return [a for a in self.args if is_read(a)]

    def outputs(self):
        return [a for a in self.args if is_written(a)]


JSON_FIELDS = ("name", "signature", "args", "grid", "grid_rule", "graph_prefetch", "on_chip", "prefetch",
               "parallel_axes", "extra")
"""What a JSON spec carries: what the pipeline reads of a kernel that the Triton IR does not hold --
the signature's names and types, each tensor's role, dtype and shape, the grid, the rule, the
prefetch, and the arguments the graph keeps on chip or copies ahead. No kernel, reference or inputs:
a JSON spec is compiled, not checked."""


_TTIR_TOKEN = {"f64": "fp64", "f32": "fp32", "f16": "fp16", "bf16": "bf16", "i64": "i64", "i32": "i32",
               "i16": "i16", "i8": "i8", "i1": "i1"}
_TTIR_DTYPE = {"fp64": "float64", "fp32": "float32", "fp16": "float16", "bf16": "bfloat16", "i64": "int64",
               "i32": "int32", "i16": "int16", "i8": "int8", "i1": "bool"}


def signature_from_ttir(ttir):
    """(name, signature) of the Triton IR file `ttir`'s kernel: its function's symbol and each parameter's
    Triton type token by name, in order. A constexpr is no parameter of it, and no kernel parameter."""
    import re
    with open(ttir) as f:
        head = next((line for line in f if "tt.func" in line), None)
    if head is None:
        raise ValueError(f"{ttir} holds no tt.func")
    name = re.search(r"@(\w+)\(", head).group(1)
    signature = {}
    for m in re.finditer(r"%(\w+): (?:!tt\.ptr<(\w+)>|(\w+))", head):
        elem, scalar = m.group(2), m.group(3)
        token = _TTIR_TOKEN.get(elem or scalar)
        if token is None:
            raise ValueError(f"{ttir}: parameter {m.group(1)} has the type {elem or scalar}, which has no Triton token")
        signature[m.group(1)] = ("*" + token) if elem else token
    return name, signature


def spec_from_json(text, ttir=""):
    """A KernelSpec from JSON_FIELDS' JSON; `ttir` is the path of the Triton IR it compiles. A spec that
    leaves out `name`, `signature` or an argument's `name` and `dtype` has them read off the IR's function."""
    import json
    got = json.loads(text)
    unknown = sorted(set(got) - set(JSON_FIELDS))
    if unknown:
        raise ValueError(f"a JSON spec holds {', '.join(JSON_FIELDS)}, not {', '.join(unknown)}")
    if not got.get("name") or not got.get("signature"):
        name, signature = signature_from_ttir(ttir)
        got["name"] = got.get("name") or name
        got["signature"] = got.get("signature") or signature
    pointers = [(n, _TTIR_DTYPE[t[1:]]) for n, t in got["signature"].items() if t.startswith("*")]
    wanted = got.pop("args")
    if len(wanted) != len(pointers):
        raise ValueError(f"{got['name']}: the spec states {len(wanted)} tensors, its kernel takes {len(pointers)}")
    args = [Arg(a.get("name") or n, a["role"], a.get("dtype") or d, tuple(a["shape"]))
            for a, (n, d) in zip(wanted, pointers)]
    got["grid"] = tuple(got["grid"])
    if got.get("parallel_axes") is not None:
        got["parallel_axes"] = tuple(got["parallel_axes"])
    return KernelSpec(kernel=None, constexprs={}, args=args, reference=None, make_inputs=None,
                      ttir=ttir, **got)


def load_spec(path):
    """Import a kernel file and return its SPEC; a `.json` file is a JSON spec, compiled from the
    Triton IR beside it (`<name>.ttir`)."""
    import importlib.util
    import os
    if path.endswith(".json"):
        with open(path) as f:
            return spec_from_json(f.read(), os.path.splitext(os.path.abspath(path))[0] + ".ttir")

    spec_mod = importlib.util.spec_from_file_location(
        "pytorchsim_kernel_" + os.path.basename(path).replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    if not hasattr(mod, "SPEC"):
        raise SystemExit(f"{path}: kernel file must define a module-level SPEC (KernelSpec)")
    return mod.SPEC
