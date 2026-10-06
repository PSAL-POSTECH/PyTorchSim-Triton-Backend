"""The compiler package."""
import os as _os
import sys as _sys

from . import machine
from . import toolchain as _config

_config.activate_bindings(_sys.path)

# Stage 1 runs in its own process and touches no MLIR. Registering there would
# stand a second MLIR beside triton's own, and the two registries cross -- which
# is what an EXT=ON libtriton turns from latent into fatal.
if not _os.environ.get("TORCHSIM_COMPILE_TTIR_ONLY"):
    from .dialects import torchsim as _torchsim

    _torchsim.register()
