"""The driver Triton requires beside every backend."""
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class CPUDriver(DriverBase):

    @staticmethod
    def is_active():
        return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        raise NotImplementedError("the torchsim backend launches nothing")

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def get_active_torch_device(self):
        raise NotImplementedError("the torchsim backend launches nothing")

    def get_benchmarker(self):
        raise NotImplementedError("the torchsim backend launches nothing")
