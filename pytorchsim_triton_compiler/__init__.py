"""The compiler package. Importing it only puts LLVM_BIN's MLIR bindings first on the path; the
pytorchsim dialect registers itself when `dialect.pytorchsim` is first imported, so `doctor` can
report a missing or stale dialect build instead of dying on it."""
import sys as _sys

from .driver import machine
from . import toolchain as _config

_config.activate_bindings(_sys.path)
