"""The LLVM the TOG passes are written against, checked when togsim/ is imported (its __init__)."""
from .. import toolchain as _config

REQUIRED_LLVM_MAJOR = 23


def llvm_major(bin_dir=None):
    """LLVM_VERSION_MAJOR of the install holding `bin_dir`, or None if unreadable."""
    import os
    import re

    bin_dir = _config.LLVM_BIN if bin_dir is None else bin_dir
    header = os.path.join(os.path.dirname(bin_dir.rstrip("/")),
                          "include", "llvm", "Config", "llvm-toolchain.h")
    try:
        with open(header) as fh:
            m = re.search(r"define\s+LLVM_VERSION_MAJOR\s+(\d+)", fh.read())
    except OSError:
        return None
    return int(m.group(1)) if m else None


def check_llvm_version():
    """Raise if PYTORCHSIM_COMPILE_LLVM_PATH is not REQUIRED_LLVM_MAJOR."""
    got = llvm_major()
    if got is None or got == REQUIRED_LLVM_MAJOR:
        return
    raise RuntimeError(
        f"the configured LLVM is {got}, and this package is written against "
        f"LLVM {REQUIRED_LLVM_MAJOR} -- the one that prints the IR it reads.\n"
        f"  configured: {_config.LLVM_BIN}\n"
        f"  `pytorchsim-triton-compiler doctor` reports what every side holds.")


check_llvm_version()


_config.assert_bindings_clean()
