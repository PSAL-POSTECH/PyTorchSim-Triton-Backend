"""A target is its kind: lowering/ext/<family>/ states KIND, an instance of a TargetKind subclass, which the
passes ask through target_kind() and nothing else."""
import abc
import importlib

from ...contract.target import ACCELERATORS, KINDS, TargetError, is_accel, kind_of

EXT_PACKAGE = __name__.rsplit(".", 2)[0] + ".ext"


class PassGateError(TargetError):
    """The target's kind cannot run the pipelines as declared."""


class TargetKind(abc.ABC):
    """What a target kind answers. A new kind writes the abstract members, data movement, which every kernel
    needs, and is refused when it is made without one; every other member has its default here, and an op a
    default leaves is refused by the pass that asks it (its FORBIDS)."""

    # Data movement: the DMA and the descriptor it encodes.
    @property
    @abc.abstractmethod
    def DMA_DIMS(self):
        """The axes one descriptor walks; legalize-transfers cuts a transfer to this many."""

    @property
    @abc.abstractmethod
    def INDEX_KEY(self):
        """The func7 of the instruction that names the program to the model."""

    @abc.abstractmethod
    def lower_transfers(self, module, transfers, vectorlane):
        """The kind's DMA for each pytorchsim.transfer of `transfers`, written before it (the pass drops)."""

    @abc.abstractmethod
    def insn(self, func7, rs1, rs2):
        """The DMA instruction as inline asm spells it."""

    @abc.abstractmethod
    def walk(self, op, vectorlane):
        """(dims, DRAM strides, lead) a transfer's descriptor walks: what TOGSim times."""

    @abc.abstractmethod
    def sram_footprints(self, op, vectorlane, interval_of):
        """What a transfer touches in each lane's bank, as its descriptor states it."""

    def lower_wait(self, op):
        """The kind's wait for pytorchsim.wait `op`, written before it (the pass drops it); none here."""

    # A unit's ops. TO_VCIX maps each unit op the kind's lowerings make to its encoder, fn(op), which replaces
    # it by its VCIX instruction.
    TO_VCIX = {}

    def lower_convert(self, module, **opts):
        """Every pytorchsim.convert_layout onto its carrier: here each bank's copy from its own store."""
        from ..ext.rvv_kind import lower_convert
        return lower_convert.run(module, **opts)

    def convert_to_target_kind(self, module, **opts):
        """Every pytorchsim op the kind's units compute -> its own ops; none here."""
        return module

    @property
    def BANK_VIEW(self):
        """The op a lane's bank view of a scratchpad tile is minted as (lib_bank_layout.bank_view): here
        RVV's."""
        from ...dialects.pytorchsim_rvv import BANK_VIEW
        return BANK_VIEW

    # Speed.
    CROSS_LANE_FOLDS = {}
    CROSS_LANE_FOLD_TYPE = ""

    def max_lmul(self, loop):
        """The widest LMUL `loop` may take for what it computes; None: the kernel's."""
        return None

    def schedule_matmul(self, module, **opts):
        """Where the matrix unit's ops are issued; here where convert_to_target_kind put them."""
        return module

    def schedule_vpu(self, module, **opts):
        """Where the vector loops' unit ops are placed; here in trip order."""
        return module

    # Resident operands: a copy node loads a dot operand whole before the kernel, which reads it held; how
    # many the graph holds is contract/copy_node.graph_prefetches'.
    def prefetch_order(self, candidates):
        """The {argument: contract.copy_node.Candidate} arguments in the order the kind holds them: here
        the larger first."""
        return sorted(candidates, key=lambda a: -candidates[a].bytes)


def family_of(target):
    """The package under lowering/ext/ that lowers `target`'s kind (Target.kind: the machine file's `kind`,
    else its built-in accelerator's, contract.target.ACCELERATORS)."""
    accel = next((e for e in target.isa if is_accel(e)), None)
    if accel is None:
        raise PassGateError(f"{target.name}'s isa ({', '.join(target.isa)}) names no accelerator "
                            f"(pytorchsim_<name>), so no kind lowers for it")
    if target.kind is None:
        raise PassGateError(f"{target.name}'s accelerator {accel} states no kind and has no family built in: "
                            f"ACCELERATORS knows {', '.join(ACCELERATORS)}")
    package = kind_of(target)
    if package is None:
        raise PassGateError(f"{target.name}'s kind {target.kind!r} is none the compiler lowers: kinds "
                            f"{', '.join(KINDS)}")
    return package


_KINDS = {}


def target_kind(target=None):
    """`target`'s kind (the machine's, unsaid): its family's KIND, loaded once."""
    if target is None:
        from ... import machine
        target = machine.target()
    family = family_of(target)
    if family not in _KINDS:
        kind = getattr(importlib.import_module(f"{EXT_PACKAGE}.{family}"), "KIND", None)
        if not isinstance(kind, TargetKind):
            raise PassGateError(f"lowering/ext/{family} states no KIND, an instance of a TargetKind subclass")
        _KINDS[family] = kind
    return _KINDS[family]
