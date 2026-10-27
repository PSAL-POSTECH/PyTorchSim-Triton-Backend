"""What a target answers (Target), the base of the one class each target's backend/compiler.py defines, as a
Triton backend's compiler.py defines its BaseBackend subclass; the passes ask it through targets.target()."""
import abc

from ..contract.target import TargetError


class PassGateError(TargetError):
    """The subtarget's target cannot run the pipelines as declared."""


class Target(abc.ABC):
    """What a target answers. A new target writes the abstract members, its dialect, data movement and the
    passes that carry its layout conversions, which every kernel needs, and is refused when it is made
    without one; every other member has its default here, and an op a default leaves is refused where the
    target's passes end (passes.TargetPasses)."""

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
        """The target's DMA for each pytorchsim.transfer of `transfers`, written before it (the pass
        drops it)."""

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
        """The target's wait for pytorchsim.wait `op`, written before it (the pass drops it); none here."""

    # A unit's ops. TO_VCIX maps each unit op the target's lowerings make to its encoder, fn(op), which
    # replaces it by its VCIX instruction.
    TO_VCIX = {}

    # Layer B: the target's own passes, each a module with run(module, **opts), in the order they run.
    @property
    @abc.abstractmethod
    def TRANSFORM_PASSES(self):
        """The passes transform runs after the common ones: every pytorchsim.convert_layout onto its
        carrier."""

    LOWER_PASSES = ()
    """The passes lower runs before the common ones: every pytorchsim op the target's units compute -> its
    own ops, and where they are issued; none here."""

    @property
    @abc.abstractmethod
    def DIALECT(self):
        """The target's dialect, which its CMakeLists.txt registers with the build, and which keys what
        the C++ asks of the target (its lane reduces)."""

    # Speed.

    def max_lmul(self, loop):
        """The widest LMUL `loop` may take for what it computes; None: the kernel's."""
        return None

    # Resident operands: a copy node loads a dot operand whole before the kernel, which reads it held; how
    # many the graph holds is contract/copy_node.graph_prefetches'.
    def prefetch_order(self, candidates):
        """The {argument: contract.copy_node.Candidate} arguments in the order the target holds them: here
        the larger first."""
        return sorted(candidates, key=lambda a: -candidates[a].bytes)
