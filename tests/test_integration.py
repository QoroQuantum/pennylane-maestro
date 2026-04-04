import pytest
import pennylane as qml
from pennylane import numpy as np

def test_gradient_parameter_shift(tol):
    """Test parameter-shift rules work correctly with the Maestro device."""
    dev_maestro = qml.device("maestro.qubit", wires=1)
    dev_default = qml.device("default.qubit", wires=1)

    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    qnode_maestro = qml.QNode(circuit, dev_maestro, diff_method="parameter-shift")
    qnode_default = qml.QNode(circuit, dev_default)

    x = np.array(0.543, requires_grad=True)
    
    grad_maestro = qml.grad(qnode_maestro)(x)
    grad_default = qml.grad(qnode_default)(x)

    assert np.isclose(grad_maestro, grad_default, **tol)


def test_batch_execution(tol):
    """Test execution of a batch of tapes."""
    dev = qml.device("maestro.qubit", wires=1)

    tape1 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
    tape2 = qml.tape.QuantumScript([qml.RX(0.2, wires=0)], [qml.expval(qml.PauliZ(0))])

    results = dev.execute((tape1, tape2))

    assert len(results) == 2
    assert np.isclose(results[0], np.cos(0.1), **tol)
    assert np.isclose(results[1], np.cos(0.2), **tol)


def test_bell_state_end_to_end(tol, shots):
    """Test a 2-qubit Bell state circuit."""
    dev = qml.device("maestro.qubit", wires=2, shots=shots)

    @qml.qnode(dev)
    def bell():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    probs = bell()
    expected = [0.5, 0.0, 0.0, 0.5]

    if shots is None:
        assert np.allclose(probs, expected, **tol)
    else:
        assert np.allclose(probs, expected, atol=0.05)


def test_ghz_state_end_to_end(tol, shots):
    """Test a 3-qubit GHZ state circuit."""
    dev = qml.device("maestro.qubit", wires=3, shots=shots)

    @qml.qnode(dev)
    def ghz():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliZ(1) @ qml.PauliZ(2))

    res = ghz()
    
    if shots is None:
        assert np.isclose(res[0], 1.0, **tol)
        assert np.isclose(res[1], 1.0, **tol)
    else:
        assert np.isclose(res[0], 1.0, atol=0.05)
        assert np.isclose(res[1], 1.0, atol=0.05)


def test_mps_mode(tol, shots):
    """Test execution using the Matrix Product State simulation mode."""
    dev = qml.device(
        "maestro.qubit",
        wires=4,
        shots=shots,
        simulation_type="MatrixProductState"
    )

    @qml.qnode(dev)
    def weak_entanglement():
        for i in range(4):
            qml.Hadamard(wires=i)
        qml.CZ(wires=[0, 1])
        qml.CZ(wires=[2, 3])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    # Analytically, CZ on |++> creates |+>|0>-|->|1>/sqrt(2).
    # Both Z0 and Z1 expectation values on this state are 0.
    res = weak_entanglement()
    if shots is None:
        assert np.isclose(res, 0.0, **tol)
    else:
        assert np.isclose(res, 0.0, atol=0.1)

def test_decomposition(tol):
    """Test that unsupported gates are decomposed correctly."""
    dev_maestro = qml.device("maestro.qubit", wires=1)
    dev_default = qml.device("default.qubit", wires=1)

    def circuit():
        # qml.Rot is not in the maestro native gates map
        qml.Rot(0.1, 0.2, 0.3, wires=0)
        return qml.expval(qml.PauliZ(0))

    qnode_maestro = qml.QNode(circuit, dev_maestro)
    qnode_default = qml.QNode(circuit, dev_default)

    assert np.isclose(qnode_maestro(), qnode_default(), **tol)
