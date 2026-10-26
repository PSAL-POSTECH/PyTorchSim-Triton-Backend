"""Spike -- the functional device: the package's launcher (runtime/spike.py) run as a stage, with
this suite's Spike, pk, job count, process spawner and failure spelling."""
import os

from pytorchsim_triton_compiler.contract import kernel_object
from pytorchsim_triton_compiler.runtime import spike
from . import as_compiled, config
from .base import Device


class SpikeDevice(Device):
    """Runs the whole launch grid, splitting it across processes when it can."""

    name = "spike"

    def launch(self, req):
        return _launch(req)


def _launch(req):
    """Spike over req's tensors under the options the object was compiled for; a failed run's message
    names the trap its exit code means. PYTORCHSIM_COMPILE_SPIKE_JOBS overrides the job count."""
    elf = req.artifact("elf")
    run = as_compiled.run_options(req, kernel_object.SPIKE_OPTIONS)
    try:
        with req.step("run the grid"):
            spike.launch(elf, req.manifest, req.runtime_dir, req.raw_paths, run,
                         spike=config.SPIKE, pk=config.PK, log=req.log,
                         run=lambda cmd, cwd: req.run("spike", cmd, cwd=cwd), error=req.error,
                         want_jobs=os.environ.get("PYTORCHSIM_COMPILE_SPIKE_JOBS", "").strip())
    except Exception as e:
        spike.name_trap(e)
        raise
    req.log("[spike] spike finished")
    return req.runtime_dir
