"""The vcix-accelerator model library an accelerator extension vcix_<name> runs on: among a build's
*.so, the one whose vcix_accel_model() descriptor states this ABI and the name <name>, not a file name.
The descriptor starts uint32 abi_version, const char *name (include/vcix_accel.h). And the gem5 machine
the model is timed on: examples/<name>/gem5/script_systolic.py of the vcix-accelerator checkout."""
import ctypes
import glob
import os

from .target import ACCEL_PREFIX

# VCIX_ACCEL_ABI_VERSION of the vcix-accelerator third_party/ pins.
ABI_VERSION = 11

# The vcix-accelerator checkout the compiler carries.
VCIX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "third_party", "vcix-accelerator")


class ModelError(RuntimeError):
    pass


class MachineError(ModelError):
    """The model has no gem5 machine in the vcix-accelerator checkout."""


def gem5_script(accel):
    """The gem5 machine `accel` (vcix_<name>) is timed on: the checkout's
    examples/<name>/gem5/script_systolic.py, each model its own. A model without one is refused, naming
    the path; no other model's machine stands in for it."""
    name = accel[len(ACCEL_PREFIX):]
    path = os.path.join(VCIX_DIR, "examples", name, "gem5", "script_systolic.py")
    if not os.path.isfile(path):
        raise MachineError(f"no gem5 machine for the vcix model {name}: {path} does not exist")
    return path


class _Head(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("name", ctypes.c_char_p)]


def describe(path):
    """(abi_version, name) the library at `path` states, name None at another ABI; None when it is
    no model."""
    try:
        lib = ctypes.CDLL(os.path.abspath(path), mode=os.RTLD_LOCAL)
        entry = lib.vcix_accel_model
    except (OSError, AttributeError):
        return None
    entry.restype, entry.argtypes = ctypes.POINTER(_Head), []
    head = entry()
    if not head:
        return None
    abi = head.contents.abi_version
    if abi != ABI_VERSION:
        return abi, None
    return abi, head.contents.name.decode()


def find(build_dir, accel):
    """The path of the one model under `build_dir` that `accel` (vcix_<name>) names; refuses none
    or several, naming what is there."""
    want = accel[len(ACCEL_PREFIX):]
    found, have, other_abi = [], [], []
    for path in sorted(glob.glob(os.path.join(build_dir, "*.so"))):
        got = describe(path)
        if got is None:
            continue
        abi, name = got
        if name is None:
            other_abi.append(f"{os.path.basename(path)} (ABI {abi})")
            continue
        have.append(name)
        if name == want:
            found.append(path)
    if len(found) == 1:
        return found[0]
    if found:
        raise ModelError(f"several vcix models named {want} under {build_dir}: "
                         f"{', '.join(os.path.basename(p) for p in found)}")
    raise ModelError(f"no vcix model named {want} under {build_dir}; have: "
                     f"{', '.join(sorted(have)) or 'none'}"
                     + (f"; at another ABI than {ABI_VERSION}: {', '.join(other_abi)}"
                        if other_abi else ""))
