"""What a device is: something that runs a kernel object over tensors on disk.

The manifest says what to launch and the target says on what machine, so a
device reads neither the compiler's source nor the kernel's Python spec.
"""
import contextlib

from pytorchsim_triton_compiler.contract import kernel_object


class DeviceError(RuntimeError):
    pass


@contextlib.contextmanager
def _no_step(_name):
    """A caller that is not timing anything still has to be given a span."""
    yield


def _plain_error(message, **kw):
    return DeviceError(message)


class Launch:
    """One launch: a compiled object, where its tensors are, and how to run.

    `log` and `run` are supplied by the caller so a device never has to know how
    the thing driving it reports or spawns processes.
    """

    def __init__(self, obj_dir, manifest, runtime_dir, raw_paths,
                 log, run, step=None, error=None):
        self.obj_dir = obj_dir
        self.manifest = manifest
        self.runtime_dir = runtime_dir
        self.raw_paths = raw_paths          # arg name -> .raw path
        self.log = log
        self.run = run                      # (stage, cmd, **kw) -> CompletedProcess
        self.step = step or _no_step        # a timing span, if the caller keeps one
        self.error = error or _plain_error   # how the caller spells a failure

    def artifact(self, kind):
        """One compiler output by the name the manifest gives it."""
        try:
            return kernel_object.artifact(self.obj_dir, kind, manifest=self.manifest)
        except kernel_object.ObjectError as e:
            raise DeviceError(f"{e}. This device needs it.") from None


class Device:
    """A thing that can run a kernel object. Subclasses state `name`."""

    name = None

    def launch(self, req):
        """Run `req`, leaving each written argument in its .raw file."""
        raise NotImplementedError
