import pytest
import pennylane as qml
import numpy as np

def test_expval(tol, shots):
    """Test expectation value of PauliZ."""
    dev_maestro = qml.device("maestro.qubit", wires=1, shots=shots)
    dev_default = qml.device("default.qubit", wires=1, shots=shots)

    def circuit():
        qml.RX(np.pi/3, wires=0)
        return qml.expval(qml.PauliZ(0))

    qnode_maestro = qml.QNode(circuit, dev_maestro)
    qnode_default = qml.QNode(circuit, dev_default)

    if shots is None:
        assert np.isclose(qnode_maestro(), qnode_default(), **tol)
    else:
        assert np.isclose(qnode_maestro(), qnode_default(), atol=0.05)


def test_var(tol, shots):
    """Test variance measurement."""
    dev_maestro = qml.device("maestro.qubit", wires=1, shots=shots)
    dev_default = qml.device("default.qubit", wires=1, shots=shots)

    def circuit():
        qml.RX(np.pi/3, wires=0)
        return qml.var(qml.PauliZ(0))

    qnode_maestro = qml.QNode(circuit, dev_maestro)
    qnode_default = qml.QNode(circuit, dev_default)

    if shots is None:
        assert np.isclose(qnode_maestro(), qnode_default(), **tol)
    else:
        assert np.isclose(qnode_maestro(), qnode_default(), atol=0.05)


def test_probs(tol, shots):
    """Test probability measurement."""
    dev_maestro = qml.device("maestro.qubit", wires=2, shots=shots)
    dev_default = qml.device("default.qubit", wires=2, shots=shots)

    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    qnode_maestro = qml.QNode(circuit, dev_maestro)
    qnode_default = qml.QNode(circuit, dev_default)

    if shots is None:
        assert np.allclose(qnode_maestro(), qnode_default(), **tol)
    else:
        assert np.allclose(qnode_maestro(), qnode_default(), atol=0.05)


def test_state(tol):
    """Test statevector representation (analytic only)."""
    dev_maestro = qml.device("maestro.qubit", wires=2)
    dev_default = qml.device("default.qubit", wires=2)

    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    qnode_maestro = qml.QNode(circuit, dev_maestro)
    qnode_default = qml.QNode(circuit, dev_default)

    # Maestro and default.qubit should agree on qubit ordering and amplitudes
    assert np.allclose(qnode_maestro(), qnode_default(), **tol)


def test_sample():
    """Test sample measurement (finite shots only)."""
    dev_maestro = qml.device("maestro.qubit", wires=2, shots=100)

    @qml.qnode(dev_maestro)
    def circuit():
        qml.PauliX(wires=0)
        return qml.sample()

    samples = circuit()
    # 100 shots of 2 wires -> shape (100, 2)
    assert samples.shape == (100, 2)
    # Wire 0 should be 1, wire 1 should be 0
    assert np.all(samples[:, 0] == 1)
    assert np.all(samples[:, 1] == 0)


def test_counts():
    """Test counts measurement (finite shots only)."""
    dev_maestro = qml.device("maestro.qubit", wires=2, shots=100)

    @qml.qnode(dev_maestro)
    def circuit():
        qml.PauliX(wires=1)
        return qml.counts()

    counts = circuit()
    assert isinstance(counts, dict)
    assert "01" in counts
    assert counts["01"] == 100


def test_multiple_measurements(tol, shots):
    """Test returning multiple observables."""
    dev_maestro = qml.device("maestro.qubit", wires=2, shots=shots)
    dev_default = qml.device("default.qubit", wires=2, shots=shots)

    def circuit():
        qml.Hadamard(wires=0)
        qml.RX(np.pi/4, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

    qnode_maestro = qml.QNode(circuit, dev_maestro)
    qnode_default = qml.QNode(circuit, dev_default)

    res_maestro = qnode_maestro()
    res_default = qnode_default()

    assert isinstance(res_maestro, tuple)
    assert len(res_maestro) == 2

    if shots is None:
        assert np.isclose(res_maestro[0], res_default[0], **tol)
        assert np.isclose(res_maestro[1], res_default[1], **tol)
    else:
        assert np.isclose(res_maestro[0], res_default[0], atol=0.05)
        assert np.isclose(res_maestro[1], res_default[1], atol=0.05)
