"""What a launch runs under: the kernel object's `run` and target. An override is logged as not
as compiled."""
from pytorchsim_triton_compiler.contract import target as _target
from pytorchsim_triton_compiler.contract import vcix_model
from . import config


def target(req):
    """The machine the object was compiled for: the file its manifest names, refused when that file
    states otherwise than the manifest recorded."""
    try:
        return _target.as_compiled(req.manifest.get("target"), f"{req.obj_dir}: manifest target")
    except _target.TargetError as e:
        raise req.error(str(e))


def _overridden(req, what, env, value):
    req.log(f"[as-compiled] {what} is {value} from {env}: not as compiled")
    return value


def spike_isa(req):
    """Spike's --isa."""
    if config.SPIKE_ISA:
        return _overridden(req, "the ISA", "TORCHSIM_COMPILE_SPIKE_ISA", config.SPIKE_ISA)
    return req.manifest["run"]["spike_isa"]


def memory(req):
    """Spike's -m: the regions the object names, as Spike spells them."""
    return ",".join(f"0x{base:x}:0x{size:x}" for base, size in req.manifest["run"]["memory"])


def model(req):
    """The model library Spike and gem5 load: the one in the vcix build named by the object's
    accelerator; refuses when there is none."""
    if config.VCIX_MODEL:
        return _overridden(req, "the model", "TORCHSIM_COMPILE_VCIX_MODEL", config.VCIX_MODEL)
    accel = req.manifest["run"]["accel"]
    try:
        return vcix_model.find(config.VCIX_BUILD, accel)
    except vcix_model.ModelError as e:
        raise req.error(f"{e}: the object was compiled for {req.manifest['target'].get('name')}, "
                        f"whose accelerator is {accel}")


def machine_config(req):
    """The model's --machine-config: the object's machine.yml."""
    if config.MACHINE_CONFIG:
        return _overridden(req, "the machine yaml", "TORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG",
                           config.MACHINE_CONFIG)
    return req.artifact("machine_config")


def gem5_script(req):
    """The gem5 script every model is timed with."""
    return config.GEM5_SCRIPT


def machine_file(req):
    """TOGSim's --config: the machine file the object was compiled for, nothing else."""
    return target(req).file
