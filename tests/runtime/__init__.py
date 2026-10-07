"""What it takes to RUN a kernel object, which the compiler does not do.

Importing this registers `spike` and `verify` on the compiler's pipeline, so
`--to verify` exists for the suite and for nobody else. The runner underneath is
a Device reading kernel.json; PyTorchSim has its own and shares only that
manifest.
"""
from pytorchsim_triton_compiler import pipeline

from . import stages


def install():
    """Register stages 7 and 8. Idempotent."""
    pipeline.register_stage("spike", stages.stage_spike, "run on Spike")
    pipeline.register_stage("verify", stages.stage_verify,
                            "compare against the torch reference", last=True)
    pipeline.register_stage("tracecheck", stages.stage_tracecheck,
                            "replay the trace producer and check it (--tog)",
                            tog=True)
    pipeline.register_stage("timing", stages.stage_timing,
                            "gem5 tile cycles, then TOGSim on the trace (--timing)",
                            on_demand=True)
    return pipeline.STAGES


install()
