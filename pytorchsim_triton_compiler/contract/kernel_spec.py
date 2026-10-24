"""What a kernel file must declare."""
import math

from .kernel_object import ITEMSIZE
from dataclasses import dataclass, field
from typing import Callable

IN = "in"
OUT = "out"
INOUT = "inout"

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

    def __post_init__(self):
        if not self.parallel_axes:
            self.parallel_axes = ("x", "y", "z")[:max(1, len(self.grid))]
        bad = [a for a in self.parallel_axes if a not in ("x", "y", "z")]
        if bad or len(set(self.parallel_axes)) != len(self.parallel_axes):
            raise ValueError(
                f"KernelSpec.parallel_axes is a set of distinct 'x'/'y'/'z', "
                f"outermost first; got {self.parallel_axes!r}")
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


def load_spec(path):
    """Import a kernel file and return its SPEC."""
    import importlib.util
    import os

    spec_mod = importlib.util.spec_from_file_location(
        "torchsim_kernel_" + os.path.basename(path).replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    if not hasattr(mod, "SPEC"):
        raise SystemExit(f"{path}: kernel file must define a module-level SPEC (KernelSpec)")
    return mod.SPEC
