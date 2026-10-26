"""The vcix-accelerator model an accelerator pytorchsim_<name> runs on: among a build's *.so, the one whose
coprocessor_model() descriptor states this ABI and <name> (include/coprocessor.h), and its gem5 machine,
examples/<name>/gem5/script_systolic.py of the vcix-accelerator checkout."""
import ctypes
import glob
import os

from .target import ACCEL_PREFIX

# COPROCESSOR_ABI_VERSION of the vcix-accelerator third_party/ pins.
ABI_VERSION = 12

# The vcix-accelerator checkout the compiler carries.
VCIX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "third_party", "vcix-accelerator")


class ModelError(RuntimeError):
    pass


class MachineError(ModelError):
    """The model has no gem5 machine in the vcix-accelerator checkout."""


def gem5_script(accel):
    """The gem5 machine `accel` (pytorchsim_<name>) is timed on: the checkout's
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
        entry = lib.coprocessor_model
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


class _Stat(ctypes.Structure):
    _fields_ = [("unit", ctypes.c_char_p), ("name", ctypes.c_char_p), ("unit_of_work", ctypes.c_char_p),
                ("kind", ctypes.c_uint32)]


_GET = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p)


class _Config(ctypes.Structure):
    _fields_ = [("ctx", ctypes.c_void_p), ("get", _GET)]


class _Model(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("name", ctypes.c_char_p), ("encodings", ctypes.c_void_p),
                ("num_encodings", ctypes.c_size_t),
                ("create", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(_Config), ctypes.c_char_p,
                                            ctypes.c_size_t)),
                ("destroy", ctypes.CFUNCTYPE(None, ctypes.c_void_p))]
    _fields_ += [(n, ctypes.c_void_p) for n in ("execute", "can_accept", "issue", "squash", "commit", "tick",
                                                "ready", "reset")]
    _fields_ += [("num_stats", ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)),
                 ("stat", ctypes.CFUNCTYPE(ctypes.POINTER(_Stat), ctypes.c_void_p, ctypes.c_size_t))]


# VCIX_STAT_ADMITTED (include/coprocessor.h): one such entry a unit.
_ADMITTED = 0


def units(path):
    """Unit names the model .so at `path` lists in its statistics."""
    lib = ctypes.CDLL(os.path.abspath(path), mode=os.RTLD_LOCAL)
    entry = lib.coprocessor_model
    entry.restype, entry.argtypes = ctypes.POINTER(_Model), []
    m = entry().contents
    if m.abi_version != ABI_VERSION:
        raise ModelError(f"{path}: ABI {m.abi_version}, and this compiler reads {ABI_VERSION}")
    get = _GET(lambda ctx, key: None)
    error = ctypes.create_string_buffer(256)
    self = m.create(ctypes.byref(_Config(None, get)), error, len(error))
    if not self:
        raise ModelError(f"{path}: create refused: {error.value.decode()}")
    try:
        got = [m.stat(self, i).contents for i in range(m.num_stats(self))]
        return tuple(dict.fromkeys(s.unit.decode() for s in got if s.kind == _ADMITTED))
    finally:
        m.destroy(self)


def find(build_dir, accel):
    """The path of the one model under `build_dir` that `accel` (pytorchsim_<name>) names; refuses none
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
