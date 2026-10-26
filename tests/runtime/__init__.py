"""What it takes to RUN a kernel object, which the compiler does not do.

Importing this registers `spike` and `verify` on the compiler's pipeline, and `togcheck` on its
--tog branch, so `--to verify` exists for the suite and for nobody else; `install_timing` adds
`timing`, for tests/pytorchsim-compile --timing. The runner underneath is a Device reading
kernel.json; the Spike one is the package's launcher (runtime/spike.py), which PyTorchSim calls too.
"""
from pytorchsim_triton_compiler import pipeline

from . import stages


def install():
    """Register spike, verify and togcheck. Idempotent."""
    pipeline.register_stage("spike", stages.stage_spike, "run on Spike")
    pipeline.register_stage("verify", stages.stage_verify,
                            "compare against the torch reference")
    pipeline.register_stage("togcheck", stages.stage_togcheck,
                            "record the TOG with the runtime attributes spike left, and check it (--tog)",
                            tog=True, after="spike")
    return pipeline.STAGES


def install_timing():
    """Register timing right after spike: TOGSim runs the TOG on the runtime attributes the Spike run
    records, so it cannot run without one. Idempotent."""
    pipeline.register_stage("timing", stages.stage_timing,
                            "gem5 tile cycles, then TOGSim on the TOG", after="spike")
    return pipeline.STAGES


install()
