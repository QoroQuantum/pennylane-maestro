"""Tests for Hamiltonian batching, MPS modes, and numerical accuracy."""

import pytest
import pennylane as qml
from pennylane import numpy as np


class TestHamiltonianBatching:
    """Tests for the batched Hamiltonian estimation path."""

    def test_hamiltonian_expval(self, tol):
        """Test expectation value of a Hamiltonian matches default.qubit."""
        n = 6
        coeffs = [0.5] * (n - 1) + [0.3] * n
        obs = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(n - 1)] + \
              [qml.PauliX(i) for i in range(n)]
        H = qml.Hamiltonian(coeffs, obs)

        np.random.seed(42)
        params = np.random.random(n)

        def circuit():
            for i in range(n):
                qml.RY(params[i], wires=i)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(H)

        dev_m = qml.device("maestro.qubit", wires=n)
        dev_d = qml.device("default.qubit", wires=n)

        res_m = qml.QNode(circuit, dev_m)()
        res_d = qml.QNode(circuit, dev_d)()

        assert np.isclose(res_m, res_d, **tol)

    def test_hamiltonian_gradient(self, tol):
        """Test gradient through a Hamiltonian matches default.qubit."""
        n = 4
        H = qml.Hamiltonian(
            [1.0] * (n - 1),
            [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(n - 1)]
        )

        params = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)

        def circuit(p):
            for i in range(n):
                qml.RY(p[i], wires=i)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(H)

        dev_m = qml.device("maestro.qubit", wires=n)
        dev_d = qml.device("default.qubit", wires=n)

        grad_m = qml.grad(qml.QNode(circuit, dev_m, diff_method="parameter-shift"))(params)
        grad_d = qml.grad(qml.QNode(circuit, dev_d))(params)

        assert np.allclose(grad_m, grad_d, atol=1e-4)

    def test_multiple_hamiltonian_expvals(self, tol):
        """Test multiple Hamiltonian expvals in a single circuit."""
        H1 = qml.Hamiltonian([0.5, 0.5], [qml.PauliZ(0), qml.PauliZ(1)])
        H2 = qml.Hamiltonian([1.0], [qml.PauliX(0) @ qml.PauliX(1)])

        dev_m = qml.device("maestro.qubit", wires=2)
        dev_d = qml.device("default.qubit", wires=2)

        def circuit():
            qml.RY(0.5, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(H1), qml.expval(H2)

        res_m = qml.QNode(circuit, dev_m)()
        res_d = qml.QNode(circuit, dev_d)()

        assert np.allclose(res_m, res_d, **tol)


class TestMPSExecution:
    """Tests for the MPS simulation mode."""

    def test_mps_analytic_accuracy(self, tol):
        """Test MPS matches statevector for a simple circuit."""
        dev_sv = qml.device("maestro.qubit", wires=6)
        dev_mps = qml.device("maestro.qubit", wires=6,
                             simulation_type="MatrixProductState",
                             max_bond_dimension=64)

        def circuit():
            for i in range(6):
                qml.Hadamard(wires=i)
            for i in range(5):
                qml.CZ(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        res_sv = qml.QNode(circuit, dev_sv)()
        res_mps = qml.QNode(circuit, dev_mps)()

        assert np.isclose(res_sv, res_mps, **tol)

    def test_mps_large_qubit_count(self):
        """Test MPS can handle large qubit counts that statevector cannot."""
        dev = qml.device("maestro.qubit", wires=50,
                         simulation_type="MatrixProductState",
                         max_bond_dimension=32)

        @qml.qnode(dev)
        def circuit():
            for i in range(50):
                qml.Hadamard(wires=i)
            for i in range(49):
                qml.CZ(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        res = circuit()
        assert isinstance(res, (float, np.floating))

    def test_mps_shots(self):
        """Test MPS mode with finite shots produces valid counts."""
        dev = qml.device("maestro.qubit", wires=4, shots=1000,
                         simulation_type="MatrixProductState",
                         max_bond_dimension=16)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

        counts = circuit()
        total = sum(counts.values())
        assert total == 1000
        # Bell state: should only see 0000 and 1100
        for key in counts:
            assert key[:2] in ("00", "11")


class TestNumericalAccuracy:
    """Cross-validate against default.qubit for various circuits."""

    @pytest.mark.parametrize("n_wires", [1, 3, 5])
    def test_random_circuit_accuracy(self, tol, n_wires):
        """Test a random parametrized circuit matches default.qubit."""
        np.random.seed(42 + n_wires)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_wires)
        params = np.random.random(shape)

        def circuit(w):
            qml.StronglyEntanglingLayers(w, wires=range(n_wires))
            return qml.expval(qml.PauliZ(0))

        dev_m = qml.device("maestro.qubit", wires=n_wires)
        dev_d = qml.device("default.qubit", wires=n_wires)

        res_m = qml.QNode(circuit, dev_m)(params)
        res_d = qml.QNode(circuit, dev_d)(params)

        assert np.isclose(res_m, res_d, **tol)

    def test_multi_qubit_observable(self, tol):
        """Test a tensor product observable (ZZZ)."""
        dev_m = qml.device("maestro.qubit", wires=3)
        dev_d = qml.device("default.qubit", wires=3)

        def circuit():
            qml.RX(0.5, wires=0)
            qml.RY(0.3, wires=1)
            qml.RZ(0.7, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        res_m = qml.QNode(circuit, dev_m)()
        res_d = qml.QNode(circuit, dev_d)()

        assert np.isclose(res_m, res_d, **tol)

    def test_sprod_observable(self, tol):
        """Test scalar-product observable (0.5 * Z)."""
        dev_m = qml.device("maestro.qubit", wires=1)
        dev_d = qml.device("default.qubit", wires=1)

        def circuit():
            qml.RX(0.7, wires=0)
            return qml.expval(0.5 * qml.PauliZ(0))

        res_m = qml.QNode(circuit, dev_m)()
        res_d = qml.QNode(circuit, dev_d)()

        assert np.isclose(res_m, res_d, **tol)

    def test_identity_observable(self, tol):
        """Test identity observable returns 1.0."""
        dev = qml.device("maestro.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.Identity(0))

        assert np.isclose(circuit(), 1.0, **tol)

    def test_variance_accuracy(self, tol):
        """Test variance calculation matches default.qubit."""
        dev_m = qml.device("maestro.qubit", wires=1)
        dev_d = qml.device("default.qubit", wires=1)

        def circuit():
            qml.RX(0.543, wires=0)
            return qml.var(qml.PauliZ(0))

        res_m = qml.QNode(circuit, dev_m)()
        res_d = qml.QNode(circuit, dev_d)()

        assert np.isclose(res_m, res_d, **tol)
