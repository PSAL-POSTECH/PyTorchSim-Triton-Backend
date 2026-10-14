"""What a launch runs under: the kernel object's `run` and target. An override is logged as not
as compiled; the TOGSim config is checked against the target either way."""
import os

from pytorchsim_triton_compiler.contract import target as _target
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
    """The accelerator model library Spike and gem5 load; refuses one that is not built."""
    if config.VCIX_MODEL:
        path = _overridden(req, "the model", "TORCHSIM_COMPILE_VCIX_MODEL", config.VCIX_MODEL)
    else:
        path = os.path.join(config.VCIX_BUILD, req.manifest["run"]["model"])
    if not os.path.isfile(path):
        unit = req.manifest["target"].get("matmul_unit", _target.DEFAULT_MATMUL_UNIT)
        raise req.error(f"the accelerator model {path} is not built: the object was compiled for "
                        f"{req.manifest['target'].get('name')} (matmul_unit {unit}), which runs "
                        f"on {os.path.basename(path)} and nothing else")
    return path


def machine_config(req):
    """The model's --machine-config: the object's machine.yml."""
    if config.MACHINE_CONFIG:
        return _overridden(req, "the machine yaml", "TORCHSIM_COMPILE_SPIKE_MACHINE_CONFIG",
                           config.MACHINE_CONFIG)
    return req.artifact("machine_config")


def gem5_script(req):
    """The gem5 script for this object's matmul unit; refuses when there is none."""
    script = req.manifest["run"]["gem5_script"]
    if script is None:
        raise req.error(f"no gem5 script times {req.manifest['run']['model']}, the model "
                        f"{req.manifest['target'].get('name')} runs on")
    return os.path.join(config.GEM5_SCRIPTS, script)


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
