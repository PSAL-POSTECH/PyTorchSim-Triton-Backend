"""The driver Triton requires beside every backend. It is the active one where PyTorch has the
device this backend compiles for -- PyTorchSim's PrivateUse1 device -- so Triton kernels a model
registers (torch.library.triton_op, wrap_triton) trace against this target; the torchsim compiler
alone, with no such device, leaves it inactive. It launches nothing: the device runs the kernels."""
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


def _device_type():
    """The PrivateUse1 device's name when PyTorch has one registered, else None."""
    try:
        import torch
        name = torch._C._get_privateuse1_backend_name()
        return name if name != "privateuseone" else None
    except Exception:
        return None


class CPUDriver(DriverBase):

    @staticmethod
    def is_active():
        return _device_type() is not None

    def map_python_to_cpp_type(self, ty: str) -> str:
        raise NotImplementedError("the torchsim backend launches nothing")

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def get_active_torch_device(self):
        import torch
        return torch.device(_device_type() or "cpu", 0)

    def get_benchmarker(self):
        raise NotImplementedError("the torchsim backend launches nothing")
