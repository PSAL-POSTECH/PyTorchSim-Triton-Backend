"""What a launch runs under: the kernel object's `run` and target. An override is logged as not
as compiled; the TOGSim config is checked against the target either way."""
import os

from pytorchsim_triton_compiler.contract import target as _target
from pytorchsim_triton_compiler.contract import vcix_model
from . import config


def target(req):
    """The machine the object was compiled for."""
    return _target.parse(req.manifest["target"], f"{req.obj_dir}: manifest target")


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


def togsim_config(req):
    """The override or the target's togsim_config; refuses none, or one that disagrees with the target."""
    import yaml
    t = target(req)
    if config.TOGSIM_CONFIG:
        path = config.TOGSIM_CONFIG
    elif t.togsim_config:
        path = os.path.join(config.TOGSIM_CONFIGS, t.togsim_config)
    else:
        raise req.error(f"target {t.name} names no togsim_config, and TORCHSIM_COMPILE_TOGSIM_CONFIG "
                        f"is not set: there is no TOGSim machine to time it on")
    try:
        with open(path) as fh:
            bad = t.togsim_disagreements(yaml.safe_load(fh) or {})
    except (OSError, _target.TargetError) as e:
        raise req.error(f"TOGSim config {path}: {e}")
    if bad:
        raise req.error(f"TOGSim config {path} is not the machine {t.name} was compiled for: "
                        + "; ".join(bad))
    return path
