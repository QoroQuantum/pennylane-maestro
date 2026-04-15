"""Converts PennyLane QuantumScript tapes into Maestro QuantumCircuit objects."""

from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
from pennylane.measurements import MidMeasureMP
from pennylane.ops.op_math import Conditional
from maestro.circuits import QuantumCircuit

# ---------------------------------------------------------------------------
# Gate map: PennyLane operation name → (maestro method name, has_params)
# ---------------------------------------------------------------------------
GATE_MAP = {
    "PauliX":               ("x",     False),
    "PauliY":               ("y",     False),
    "PauliZ":               ("z",     False),
    "Hadamard":             ("h",     False),
    "S":                    ("s",     False),
    "T":                    ("t",     False),
    "SX":                   ("sx",    False),
    "RX":                   ("rx",    True),
    "RY":                   ("ry",    True),
    "RZ":                   ("rz",    True),
    "PhaseShift":           ("p",     True),
    "CNOT":                 ("cx",    False),
    "CY":                   ("cy",    False),
    "CZ":                   ("cz",    False),
    "CH":                   ("ch",    False),
    "SWAP":                 ("swap",  False),
    "Toffoli":              ("ccx",   False),
    "CSWAP":                ("cswap", False),
    "CRX":                  ("crx",   True),
    "CRY":                  ("cry",   True),
    "CRZ":                  ("crz",   True),
    "ControlledPhaseShift": ("cp",    True),
    "U3":                   ("u",     True),
}

# Adjoint gates that map to dedicated Maestro instructions.
ADJOINT_MAP = {
    "S":  "sdg",
    "T":  "tdg",
    "SX": "sxdg",
}


def _apply_operation(
    qc: QuantumCircuit,
    op: qml.operation.Operator,
    mcm_tracker: "MCMTracker | None" = None,
) -> None:
    """Apply a single PennyLane operation to a Maestro QuantumCircuit.

    Handles regular gates, adjoint wrappers, parametric gates, and
    mid-circuit measurements (with optional qubit reset).
    Wires must already be mapped to 0-indexed integers.

    Args:
        qc: The Maestro circuit to append operations to.
        op: The PennyLane operation.
        mcm_tracker: If provided, enables native MCM mode — each
            ``MidMeasureMP`` is assigned a unique classical bit index
            via this tracker.  If ``None``, the legacy deferred path
            is used.
    """
    # ── Mid-circuit measurement ───────────────────────────
    if isinstance(op, MidMeasureMP):
        wire = int(op.wires[0])
        if mcm_tracker is not None:
            # Native MCM: measure into a tracked classical bit
            bit_idx = mcm_tracker.allocate(op)
            qc.measure([(wire, bit_idx)])
            if op.reset:
                qc.reset(wire)
        else:
            # Legacy path (deferred mode): measure qubit→qubit mapping
            qc.measure([(wire, wire)])
            if op.reset:
                qc.reset(wire)
        return

    # ── Conditional (classically-controlled) op ───────────
    # After dynamic_one_shot these are resolved before reaching the
    # device. If one slips through, raise a clear error.
    if isinstance(op, Conditional):
        raise ValueError(
            "Conditional operations must be resolved before execution. "
            "Use mcm_method='one-shot' so that PennyLane's dynamic_one_shot "
            "transform resolves all Conditional ops into concrete gates."
        )

    # ── Adjoint handling ──────────────────────────────────
    if isinstance(op, qml.ops.op_math.Adjoint):
        base = op.base
        base_name = base.name

        # Gates with a native dagger instruction in Maestro
        if base_name in ADJOINT_MAP:
            maestro_method = ADJOINT_MAP[base_name]
            wires = [int(w) for w in op.wires]
            getattr(qc, maestro_method)(*wires)
            return

        # Parametric gates: adjoint negates the parameter(s)
        if base_name in GATE_MAP:
            method_name, has_params = GATE_MAP[base_name]
            wires = [int(w) for w in op.wires]
            if has_params:
                params = [-float(p) for p in base.parameters]
                getattr(qc, method_name)(*wires, *params)
            else:
                # Non-parametric gate without a native adjoint:
                # delegate to its matrix (should not normally happen
                # because PennyLane decomposes these).
                getattr(qc, method_name)(*wires)
            return

        raise ValueError(
            f"Unsupported adjoint operation: Adjoint({base_name}). "
            "This should have been decomposed during preprocessing."
        )

    # ── Regular gate ──────────────────────────────────────
    name = op.name
    if name not in GATE_MAP:
        raise ValueError(
            f"Unsupported operation: {name}. "
            "This should have been decomposed during preprocessing."
        )

    method_name, has_params = GATE_MAP[name]
    wires = [int(w) for w in op.wires]

    if has_params:
        params = [float(p) for p in op.parameters]
        getattr(qc, method_name)(*wires, *params)
    else:
        getattr(qc, method_name)(*wires)


def tape_to_maestro(
    tape: qml.tape.QuantumScript,
    num_wires: int,
) -> QuantumCircuit:
    """Convert a PennyLane tape (already mapped to standard wires) into a
    Maestro ``QuantumCircuit``.

    Args:
        tape: A ``QuantumScript`` whose wires are 0-indexed integers.
        num_wires: Total number of qubits for the circuit.

    Returns:
        A Maestro ``QuantumCircuit`` ready for execution.
    """
    qc = QuantumCircuit()

    # Ensure Maestro allocates exactly `num_wires` qubits by touching
    # every qubit with a no-op gate.  We use rz(q, 0.0) which is a true
    # identity matrix but, unlike X;X, is not optimised away by the
    # simulator's gate-cancellation pass.
    for q in range(num_wires):
        qc.rz(q, 0.0)

    for op in tape.operations:
        _apply_operation(qc, op)

    return qc


# ---------------------------------------------------------------------------
# Native MCM support
# ---------------------------------------------------------------------------

@dataclass
class MCMTracker:
    """Tracks classical bit allocation for native mid-circuit measurements.

    Each ``MidMeasureMP`` is assigned a unique classical bit index.
    The mapping from PennyLane's internal MCM id to classical bit index
    is stored so the device can extract outcomes from Maestro's counts.
    """

    _next_bit: int = 0
    # Maps MidMeasureMP.id → classical bit index
    id_to_bit: dict = field(default_factory=dict)

    def allocate(self, op: MidMeasureMP) -> int:
        """Allocate a new classical bit for a mid-circuit measurement."""
        bit_idx = self._next_bit
        self._next_bit += 1
        if op.id is not None:
            self.id_to_bit[op.id] = bit_idx
        return bit_idx

    @property
    def num_classical_bits(self) -> int:
        """Total number of classical bits allocated."""
        return self._next_bit


def tape_to_maestro_native(
    tape: qml.tape.QuantumScript,
    num_wires: int,
) -> tuple[QuantumCircuit, MCMTracker]:
    """Convert a PennyLane tape with MCMs into a Maestro circuit using
    native mid-circuit measurement (true wavefunction collapse).

    Unlike ``tape_to_maestro()``, this function does **not** require
    ``defer_measurements`` preprocessing.  Each ``MidMeasureMP`` is
    translated to Maestro's native ``measure([(qubit, classical_bit)])``
    instruction, keeping the circuit at the physical qubit count.

    Args:
        tape: A ``QuantumScript`` whose wires are 0-indexed integers.
        num_wires: Total number of qubits for the circuit.

    Returns:
        A tuple of ``(QuantumCircuit, MCMTracker)``.
    """
    qc = QuantumCircuit()
    tracker = MCMTracker()

    # Ensure Maestro allocates exactly `num_wires` qubits
    for q in range(num_wires):
        qc.rz(q, 0.0)

    for op in tape.operations:
        _apply_operation(qc, op, mcm_tracker=tracker)

    return qc, tracker


def observable_to_pauli_string(
    obs: qml.operation.Operator,
    num_wires: int,
) -> str:
    """Convert a PennyLane Pauli observable to a Maestro Pauli string.

    For example, ``qml.PauliZ(0)`` on 3 qubits → ``"ZII"``.
    ``qml.PauliZ(0) @ qml.PauliX(1)`` on 3 qubits → ``"ZXI"``.

    Returns ``None`` if the observable cannot be represented as a simple
    Pauli string (e.g. Hermitian, Hamiltonian).
    """
    pauli_map = {
        "PauliX": "X",
        "PauliY": "Y",
        "PauliZ": "Z",
        "Identity": "I",
    }

    # Start with all identity
    pauli_chars = ["I"] * num_wires

    # Handle tensor products (Prod)
    if isinstance(obs, qml.ops.Prod):
        for factor in obs.operands if hasattr(obs, "operands") else obs.obs:
            name = factor.name
            if name not in pauli_map:
                return None
            for w in factor.wires:
                pauli_chars[int(w)] = pauli_map[name]
        return "".join(pauli_chars)

    # Handle SProd (scalar * observable)
    if isinstance(obs, qml.ops.SProd):
        inner = observable_to_pauli_string(obs.base, num_wires)
        return inner  # scalar is handled separately

    # Single Pauli
    name = obs.name
    if name in pauli_map:
        for w in obs.wires:
            pauli_chars[int(w)] = pauli_map[name]
        return "".join(pauli_chars)

    return None


def decompose_hamiltonian_to_pauli_terms(
    obs: qml.operation.Operator,
    num_wires: int,
) -> list[tuple[float, str]] | None:
    """Decompose a Hamiltonian / LinearCombination / Sum into
    ``[(coefficient, pauli_string), ...]`` for batched ``estimate()``.

    Returns ``None`` if any term cannot be represented as a Pauli string.

    This allows Maestro to evaluate **all** terms in a single C++ call
    (one simulator pass) rather than PennyLane splitting them into
    separate tapes executed independently.
    """
    # Hamiltonian / LinearCombination: has .coeffs and .ops
    if hasattr(obs, "coeffs") and hasattr(obs, "ops"):
        terms = []
        for coeff, op in zip(obs.coeffs, obs.ops):
            ps = observable_to_pauli_string(op, num_wires)
            if ps is None:
                return None
            terms.append((float(coeff), ps))
        return terms

    # Sum: each operand may be an SProd or plain Pauli
    if isinstance(obs, qml.ops.Sum):
        terms = []
        for operand in obs.operands:
            if isinstance(operand, qml.ops.SProd):
                ps = observable_to_pauli_string(operand.base, num_wires)
                if ps is None:
                    return None
                terms.append((float(operand.scalar), ps))
            else:
                ps = observable_to_pauli_string(operand, num_wires)
                if ps is None:
                    return None
                terms.append((1.0, ps))
        return terms

    return None

