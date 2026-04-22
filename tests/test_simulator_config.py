"""Tests for SimulatorConfig integration (qoro-maestro ≥ 0.2.11).

Verifies that the device correctly builds a ``maestro.SimulatorConfig``
from user-provided settings and passes it through every execution path.
"""

import pytest
import numpy as np
import pennylane as qml

import maestro
from maestro import SimulatorConfig
from pennylane_maestro.maestro_device import MaestroQubitDevice


# ---------------------------------------------------------------------------
# _build_config unit tests
# ---------------------------------------------------------------------------

class TestBuildConfig:
    """Unit tests for MaestroQubitDevice._build_config()."""

    def test_default_config(self):
        """Default device produces a valid SimulatorConfig with QCSim/Statevector."""
        dev = qml.device("maestro.qubit", wires=1)
        cfg = dev._build_config()
        assert isinstance(cfg, SimulatorConfig)
        assert cfg.simulator_type == maestro.SimulatorType.QCSim
        assert cfg.simulation_type == maestro.SimulationType.Statevector
        assert cfg.use_double_precision is False

    def test_mps_config(self):
        """MPS settings are forwarded to the config."""
        dev = qml.device(
            "maestro.qubit",
            wires=4,
            simulation_type="MatrixProductState",
            max_bond_dimension=32,
            singular_value_threshold=1e-8,
        )
        cfg = dev._build_config()
        assert cfg.simulation_type == maestro.SimulationType.MatrixProductState
        assert cfg.max_bond_dimension == 32
        assert cfg.singular_value_threshold == pytest.approx(1e-8)

    def test_double_precision_config(self):
        """use_double_precision flag propagates."""
        dev = qml.device("maestro.qubit", wires=1, use_double_precision=True)
        cfg = dev._build_config()
        assert cfg.use_double_precision is True

    def test_optional_fields_unset(self):
        """When max_bond_dimension / SVD threshold are None, config uses defaults."""
        dev = qml.device("maestro.qubit", wires=1)
        cfg = dev._build_config()
        # The default SimulatorConfig should have None for these
        default = SimulatorConfig()
        assert cfg.max_bond_dimension == default.max_bond_dimension
        assert cfg.singular_value_threshold == default.singular_value_threshold


# ---------------------------------------------------------------------------
# End-to-end execution through SimulatorConfig
# ---------------------------------------------------------------------------

class TestSimulatorConfigExecution:
    """Verify every execution path uses SimulatorConfig without error."""

    def test_analytic_statevector_path(self):
        """Analytic mode with a non-Pauli observable (forces statevector path)."""
        dev = qml.device("maestro.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.probs(wires=[0, 1])

        probs = circuit()
        assert np.allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-6)

    def test_analytic_estimate_path(self):
        """Analytic Pauli expval (estimate fast path)."""
        dev = qml.device("maestro.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), -1.0, atol=1e-6)

    def test_finite_shots_execute_path(self):
        """Finite-shots sampling path."""
        dev = qml.device("maestro.qubit", wires=1, shots=5000)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.probs(wires=0)

        probs = circuit()
        # |1> with certainty
        assert probs[1] > 0.95

    def test_finite_shots_estimate_path(self):
        """Finite-shots Pauli expval (still uses estimate fast path)."""
        dev = qml.device("maestro.qubit", wires=1, shots=5000)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), -1.0, atol=1e-6)

    def test_hamiltonian_estimate_path(self):
        """Hamiltonian expval via batched estimate."""
        dev = qml.device("maestro.qubit", wires=2)
        H = qml.Hamiltonian([0.5, 0.3], [qml.PauliZ(0), qml.PauliZ(1)])

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.expval(H)

        # 0.5*(-1) + 0.3*(1) = -0.2
        assert np.isclose(circuit(), -0.2, atol=1e-6)

    def test_mps_execution(self):
        """MPS backend with bond dimension via SimulatorConfig."""
        dev = qml.device(
            "maestro.qubit",
            wires=4,
            simulation_type="MatrixProductState",
            max_bond_dimension=16,
        )

        @qml.qnode(dev)
        def circuit():
            for i in range(4):
                qml.Hadamard(i)
            qml.CNOT([0, 1])
            return qml.expval(qml.PauliZ(0))

        res = circuit()
        assert np.isclose(res, 0.0, atol=1e-6)

    def test_mps_finite_shots(self):
        """MPS with finite shots via SimulatorConfig."""
        dev = qml.device(
            "maestro.qubit",
            wires=2,
            shots=5000,
            simulation_type="MatrixProductState",
            max_bond_dimension=8,
        )

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.counts()

        counts = circuit()
        # Should be all "10"
        assert "10" in counts
        assert counts["10"] == 5000
