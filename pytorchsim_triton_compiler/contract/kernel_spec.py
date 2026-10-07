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
        if self.expect != "pass" and not self.expect_why:
            raise ValueError(
                "a kernel that is not expected to pass has to say why "
                "(expect_why): %s" % self.name)

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
