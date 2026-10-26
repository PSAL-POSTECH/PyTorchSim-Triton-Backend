"""What a launch runs under: the compiler's run_options() of the machine the kernel object was
compiled for. An override is logged as not as compiled."""
from pytorchsim_triton_compiler.contract import kernel_object as ko
from pytorchsim_triton_compiler.contract import target as _target
from pytorchsim_triton_compiler.contract import vcix_model
from . import config

_OVERRIDES = (("spike_isa", "the ISA", "PYTORCHSIM_COMPILE_SPIKE_ISA", lambda: config.SPIKE_ISA),
              ("model", "the model", "PYTORCHSIM_COMPILE_VCIX_MODEL", lambda: config.VCIX_MODEL),
              ("machine_config", "the machine yaml", "PYTORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG",
               lambda: config.MACHINE_CONFIG))


def target(req):
    """The machine the object was compiled for: the file its manifest names, refused when that file
    states otherwise than the manifest recorded."""
    try:
        return _target.as_compiled(req.manifest.get("target"), f"{req.obj_dir}: manifest target")
    except _target.TargetError as e:
        raise req.error(str(e))


def run_options(req, keys=None):
    """Spike's --isa and -m, the model library, its --machine-config and the gem5 script (or only
    `keys` of them), as the compiler answers them for the object's target; refuses when the vcix
    build has no such model."""
    t = target(req)
    instead = {}
    for key, what, env, value in _OVERRIDES:
        if value():
            req.log(f"[as-compiled] {what} is {value()} from {env}: not as compiled")
            instead[key] = value()
    try:
        return ko.run_options(t, req.obj_dir, config.VCIX_BUILD, req.manifest, keys, **instead)
    except vcix_model.ModelError as e:
        raise req.error(f"{e}: the object was compiled for {t.name}, whose accelerator is "
                        f"{t.accel}")
    except ko.ObjectError as e:
        raise req.error(str(e))


def machine_file(req):
    """TOGSim's --config: the machine file the object was compiled for, nothing else."""
    return target(req).file
