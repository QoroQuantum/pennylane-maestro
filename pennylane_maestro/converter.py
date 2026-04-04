"""Converts PennyLane QuantumScript tapes into Maestro QuantumCircuit objects."""

import numpy as np
import pennylane as qml
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


def _apply_operation(qc: QuantumCircuit, op: qml.operation.Operator) -> None:
    """Apply a single PennyLane operation to a Maestro QuantumCircuit.

    Handles regular gates, adjoint wrappers, and parametric gates.
    Wires must already be mapped to 0-indexed integers.
    """
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

