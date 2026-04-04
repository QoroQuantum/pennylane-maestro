import pytest
import pennylane as qml
import numpy as np
from pprint import pprint

from pennylane_maestro.converter import tape_to_maestro, observable_to_pauli_string

class TestConverter:
    """Tests for the QuantumTape to Maestro QuantumCircuit conversion."""

    def test_single_qubit_gates(self):
        """Test all supported non-parametric single qubit gates."""
        tape = qml.tape.QuantumScript([
            qml.PauliX(wires=0),
            qml.PauliY(wires=0),
            qml.PauliZ(wires=0),
            qml.Hadamard(wires=0),
            qml.S(wires=0),
            qml.T(wires=0),
            qml.SX(wires=0)
        ]).map_to_standard_wires()
        
        qc = tape_to_maestro(tape, 1)
        assert qc.num_qubits == 1
        # No built-in way to inspect QuantumCircuit ops easily in Python,
        # but execution tests will verify semantics. If it didn't crash,
        # the methods were successfully called.

    def test_parametric_single_qubit_gates(self):
        """Test supported parametric single qubit gates."""
        tape = qml.tape.QuantumScript([
            qml.RX(0.1, wires=0),
            qml.RY(0.2, wires=0),
            qml.RZ(0.3, wires=0),
            qml.PhaseShift(0.4, wires=0),
            qml.U3(0.5, 0.6, 0.7, wires=0)
        ]).map_to_standard_wires()
        
        qc = tape_to_maestro(tape, 1)
        assert qc.num_qubits == 1

    def test_two_qubit_gates(self):
        """Test supported two-qubit gates."""
        tape = qml.tape.QuantumScript([
            qml.CNOT(wires=[0, 1]),
            qml.CY(wires=[0, 1]),
            qml.CZ(wires=[0, 1]),
            qml.CH(wires=[0, 1]),
            qml.SWAP(wires=[0, 1])
        ]).map_to_standard_wires()
        
        qc = tape_to_maestro(tape, 2)
        assert qc.num_qubits == 2

    def test_controlled_parametric_gates(self):
        """Test supported controlled parametric gates."""
        tape = qml.tape.QuantumScript([
            qml.CRX(0.1, wires=[0, 1]),
            qml.CRY(0.2, wires=[0, 1]),
            qml.CRZ(0.3, wires=[0, 1]),
            qml.ControlledPhaseShift(0.4, wires=[0, 1])
        ]).map_to_standard_wires()
        
        qc = tape_to_maestro(tape, 2)
        assert qc.num_qubits == 2

    def test_three_qubit_gates(self):
        """Test supported three-qubit gates."""
        tape = qml.tape.QuantumScript([
            qml.Toffoli(wires=[0, 1, 2]),
            qml.CSWAP(wires=[0, 1, 2])
        ]).map_to_standard_wires()
        
        qc = tape_to_maestro(tape, 3)
        assert qc.num_qubits == 3

    def test_adjoint_operations(self):
        """Test adjoint wrapper handling."""
        tape = qml.tape.QuantumScript([
            qml.adjoint(qml.S(0)),
            qml.adjoint(qml.T(0)),
            qml.adjoint(qml.SX(0)),
            qml.adjoint(qml.RX(0.1, wires=0)),
            qml.adjoint(qml.PhaseShift(np.pi/2, wires=0))
        ]).map_to_standard_wires()
        
        qc = tape_to_maestro(tape, 1)
        assert qc.num_qubits == 1

    def test_wire_mapping(self):
        """Test that out-of-order and non-integer wires are handled if mapped correctly."""
        # Tape with weird wire labels
        tape = qml.tape.QuantumScript(
            [qml.CNOT(wires=["aux", "data"])],
            measurements=[qml.expval(qml.PauliZ("data"))]
        )
        
        # This is what happen inside execute()
        mapped_tape = tape.map_to_standard_wires()
        
        qc = tape_to_maestro(mapped_tape, 2)
        # 2 qubits from standard integer mapping
        assert qc.num_qubits == 2

    def test_observable_to_pauli_string(self):
        """Test conversion of PennyLane observables to strings for estimate()."""
        obs1 = qml.PauliZ(0)
        assert observable_to_pauli_string(obs1, 3) == "ZII"
        
        obs2 = qml.PauliX(2)
        assert observable_to_pauli_string(obs2, 3) == "IIX"
        
        obs3 = qml.PauliZ(0) @ qml.PauliX(1)
        assert observable_to_pauli_string(obs3, 3) == "ZXI"
        
        obs4 = qml.Identity(0)
        assert observable_to_pauli_string(obs4, 2) == "II"
        
        obs5 = qml.s_prod(2.5, qml.PauliZ(0))
        assert observable_to_pauli_string(obs5, 2) == "ZI"

        # Unsupported observables for estimate()
        obs6 = qml.Hadamard(0)
        assert observable_to_pauli_string(obs6, 2) is None
