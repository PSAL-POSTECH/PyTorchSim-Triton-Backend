"""The driver Triton requires beside every backend. It is the active one where PyTorch has the
device this backend compiles for -- PyTorchSim's PrivateUse1 device -- so Triton kernels a model
registers (torch.library.triton_op, wrap_triton) trace against this target; the pytorchsim compiler
alone, with no such device, leaves it inactive. It runs nothing itself: the device's launcher, which
whoever owns the device registers with set_launcher, runs the compiled kernels."""
import json
import os
import math

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class _Launcher:
    """Triton's launcher for a kernel: its call hands the registered launcher the binary, the kernel
    object, the grid and the arguments; a caller that passes every parameter, as Inductor's does, has the
    constexpr ones dropped, as a native launcher ignores them, and their values handed over in the kernel
    object's `constexprs`. A launch over extents the binary was not built for gets the binary built for them."""

    def __init__(self, src, metadata):
        self.metadata = metadata
        self.width = len(src.signature)
        self.constexpr = {i for i, ty in enumerate(src.signature.values()) if ty == "constexpr"
                          } | {k[0] for k in src.constants}
        self.params = [n for i, n in enumerate(src.signature) if i not in self.constexpr]
        names = list(src.signature)
        self.constexprs = {names[k[0]]: v for k, v in src.constants.items()
                           if len(k) == 1 and k[0] < len(names) and isinstance(v, (int, float, bool))}
        self.builds = {}
        _launcher()

    def __call__(self, grid_x, grid_y, grid_z, stream, function, packed_metadata, launch_metadata,
                 enter_hook, exit_hook, *args):
        if os.environ.get("PYTORCHSIM_COLLECT_FAILURES"):
            return
        if not self.metadata.kernel_object:
            raise RuntimeError("this kernel was compiled without pytorchsim_spec, which stops at the Triton IR: "
                               "it has no binary to launch")
        if len(args) == self.width:
            args = tuple(a for i, a in enumerate(args) if i not in self.constexpr)
        binary, kernel_object, artifacts = self._get_build_for(function, args)
        if self.constexprs:
            kernel_object = json.dumps({**json.loads(kernel_object), "constexprs": self.constexprs})
        _launcher()(binary, kernel_object, (grid_x, grid_y, grid_z), args, artifacts=artifacts)

    def _get_build_for(self, function, args):
        """(binary, kernel object, artifacts) built for this launch's extents: the compile's own when they are its
        spec's, else one built for them, once per extents."""
        own = (function, self.metadata.kernel_object, getattr(self.metadata, "artifacts", None))
        rebuild = json.loads(getattr(self.metadata, "rebuild", "") or "{}")
        if not rebuild:
            return own
        spec = json.loads(rebuild["spec"])
        spans = [_get_storage_span(a) for a in args if hasattr(a, "data_ptr")]
        stated = spec["extra"].get("scalar_values", {})
        scalars = {n: int(a) for n, a in zip(self.params, args) if n in stated and isinstance(a, int)}
        built = [math.prod(a["shape"]) for a in spec["args"]]
        if len(spans) != len(built) or (spans == built and scalars == {n: stated[n] for n in scalars}):
            return own
        key = (tuple(spans), tuple(sorted(scalars.items())))
        if key not in self.builds:
            from .compiler import CPUBackend
            elf, kernel_object, artifacts = CPUBackend.rebuild_for(rebuild, spans, scalars)
            self.builds[key] = (elf, kernel_object, artifacts)
        return self.builds[key]


def _get_storage_span(tensor):
    """The last addressable element of a tensor's layout, in elements."""
    return 1 + sum((s - 1) * t for s, t in zip(tensor.shape, tensor.stride()))


def _device_type():
    """The PrivateUse1 device's name when PyTorch has one registered, else None."""
    try:
        import torch
        name = torch._C._get_privateuse1_backend_name()
        return name if name != "privateuseone" else None
    except Exception:
        return None


_LAUNCHER = None


def set_launcher(fn):
    """Register the function that runs a compiled kernel: fn(binary, kernel_object, grid, args, artifacts=),
    the ELF bytes, kernel.json's text, the launch grid (x, y, z), the arguments the kernel was called with and
    the other files its object names, as JSON of kind -> {"name", "data" (base64)}: the machine config, and
    with pytorchsim_tog the TOG, its tile types and the cycle ELF.
    The device's owner registers it; this package imports none. None unregisters."""
    global _LAUNCHER
    _LAUNCHER = fn


def _launcher():
    if _LAUNCHER is None:
        raise RuntimeError("no launcher is registered with the pytorchsim triton backend: the owner of "
                           "the device registers one with pytorchsim_triton_backend.driver.set_launcher")
    return _LAUNCHER


class _Utils:
    """What Triton's CompiledKernel asks of a driver before a launch: the device's limits, and the
    binary loaded. The binary is its own handle -- the launcher runs it."""

    def get_device_properties(self, device):
        _launcher()
        return {"max_shared_mem": 2 ** 62}

    def load_binary(self, name, kernel, shared, device):
        _launcher()
        return kernel, kernel, 0, 0, 1

    def unload_module(self, module):
        pass


class CPUDriver(DriverBase):

    def __init__(self):
        self.utils = _Utils()
        self.launcher_cls = _Launcher

    def get_current_device(self):
        return 0

    def get_current_stream(self, device):
        return 0

    def set_current_device(self, device):
        pass

    @staticmethod
    def is_active():
        return _device_type() is not None

    def map_python_to_cpp_type(self, ty: str) -> str:
        raise NotImplementedError("the pytorchsim backend maps no C types: it launches through set_launcher")

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def get_active_torch_device(self):
        import torch
        return torch.device(_device_type() or "cpu", 0)

    def get_benchmarker(self):
        raise NotImplementedError("the pytorchsim backend benchmarks nothing itself")
