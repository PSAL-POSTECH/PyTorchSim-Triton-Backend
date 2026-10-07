"""The compiler package."""
import os as _os
import sys as _sys

from . import machine
from . import toolchain as _config

_config.activate_bindings(_sys.path)

if not _os.environ.get("TORCHSIM_COMPILE_TTIR_ONLY"):
    from .dialects import torchsim as _torchsim

    _torchsim.register()
