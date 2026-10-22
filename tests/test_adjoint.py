"""Analytic derivatives and native batching on the statevector path."""

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest

from pennylane_maestro import adjoint as _adjoint

BACKENDS = ["inner_product"] + (["engine"] if _adjoint.HAS_ENGINE_ADJOINT else [])
needs_engine = pytest.mark.skipif(
    not _adjoint.HAS_ENGINE_ADJOINT, reason="qoro-maestro build without AdjointProgram"
)

WIRES = ["a", 3, "q", -1]
ONE_PARAM = [qml.RX, qml.RY, qml.RZ, qml.PhaseShift]
TWO_WIRE = [qml.CRX, qml.CRY, qml.CRZ, qml.ControlledPhaseShift]
OBSERVABLES = [
    lambda w: qml.Z(w[0]),
    lambda w: qml.X(w[1]) @ qml.Y(w[3]),
    lambda w: 0.4 * qml.Z(w[2]) - 1.1 * qml.X(w[0]) @ qml.Z(w[1]) + 0.3 * qml.I(w[0]),
    lambda w: qml.Hamiltonian([0.5, -0.7], [qml.Y(w[1]), qml.Z(w[0]) @ qml.Z(w[3])]),
]


def circuit(weights, x, wires=WIRES, measurements=(0,)):
    """Every differentiable gate in config.toml, plus fixed gates and adjoints."""
    w = wires
    qml.AngleEmbedding(x, wires=w)
    k = 0
    for i, gate in enumerate(ONE_PARAM):
        gate(weights[k], wires=w[i])
        k += 1
    for i, gate in enumerate(TWO_WIRE):
        gate(weights[k], wires=[w[i], w[(i + 1) % 4]])
        k += 1
    qml.Rot(weights[k], weights[k + 1], weights[k + 2], wires=w[1])
    qml.U3(weights[k + 3], weights[k + 4], weights[k + 5], wires=w[2])
    k += 6
    qml.adjoint(qml.RY(weights[k], wires=w[3]))
    qml.adjoint(qml.Rot(weights[k + 1], weights[k + 2], weights[k + 3], wires=w[0]))
    qml.adjoint(qml.U3(weights[k + 4], weights[k + 5], weights[k + 6], wires=w[1]))
    k += 7
    qml.CNOT(wires=[w[0], w[2]])
    qml.Hadamard(wires=w[3])
    qml.adjoint(qml.S(wires=w[1]))
    qml.CZ(wires=[w[3], w[1]])
    qml.IsingXX(weights[k], wires=[w[2], w[3]])  # decomposed
    return [qml.expval(OBSERVABLES[m](w)) for m in measurements]


NUM_WEIGHTS = 4 + 4 + 6 + 7 + 1


def run(device, measurements, x, diff_method, **kwargs):
    dev = qml.device(device, wires=WIRES) if device == "default.qubit" else qml.device(
        device, wires=WIRES, **kwargs.pop("device_kwargs", {})
    )

    @qml.qnode(dev, diff_method=diff_method, **kwargs)
    def f(weights):
        res = circuit(weights, x, measurements=measurements)
        return res[0] if len(res) == 1 else qml.math.stack(res)

    weights = pnp.array(np.random.default_rng(3).uniform(-np.pi, np.pi, NUM_WEIGHTS), requires_grad=True)
    return f(weights), qml.jacobian(f)(weights), f


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("measurements", [(0,), (1,), (2,), (3,), (0, 2, 3)])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("device_vjp", [True, False])
def test_matches_default_qubit(backend, measurements, batched, device_vjp):
    if backend == "inner_product" and (batched or len(measurements) > 1) and device_vjp:
        pytest.skip("covered by the other combinations; inner_product is slow")
    rng = np.random.default_rng(1)
    x = pnp.array(rng.uniform(0, 1, (3, 4) if batched else 4), requires_grad=False)
    ref_val, ref_jac, _ = run("default.qubit", measurements, x, "backprop")
    val, jac, f = run(
        "maestro.qubit", measurements, x, "best", device_vjp=device_vjp,
        device_kwargs={"adjoint_backend": backend},
    )
    assert qml.math.allclose(val, ref_val, atol=1e-8)
    assert qml.math.allclose(jac, ref_jac, atol=1e-8)


@needs_engine
def test_best_resolves_to_adjoint_and_batches_natively():
    dev = qml.device("maestro.qubit", wires=4)

    @qml.qnode(dev)
    def f(weights, x):
        qml.AngleEmbedding(x, wires=range(4))
        qml.StronglyEntanglingLayers(weights, wires=range(4))
        return qml.expval(qml.Z(0))

    shape = qml.StronglyEntanglingLayers.shape(2, 4)
    weights = pnp.array(np.random.default_rng(0).normal(size=shape), requires_grad=True)
    x = pnp.array(np.random.default_rng(1).uniform(size=(8, 4)), requires_grad=False)
    assert qml.workflow.get_best_diff_method(f)(weights, x) == "adjoint"

    def cost(w):
        return pnp.sum(f(w, x) ** 2)

    with qml.Tracker(dev) as tracker:
        grad = qml.grad(cost)(weights)
    # the whole batch of 8 goes to the device in one call
    assert sum(v for k, v in tracker.totals.items() if k.endswith("batches")) == 1
    assert tracker.totals["derivatives"] == 8

    ref = qml.QNode(f.func, qml.device("default.qubit", wires=4), diff_method="backprop")
    ref_grad = qml.grad(lambda w: pnp.sum(ref(w, x) ** 2))(weights)
    assert qml.math.allclose(grad, ref_grad, atol=1e-8)


@needs_engine
def test_trainable_broadcast_parameters_are_expanded():
    dev = qml.device("maestro.qubit", wires=2)
    ref_dev = qml.device("default.qubit", wires=2)

    def f(x, y):
        qml.RX(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        return qml.expval(qml.Z(1))

    x = pnp.array([0.1, 0.4, 0.9], requires_grad=True)
    y = pnp.array(0.3, requires_grad=True)
    jac = qml.jacobian(qml.QNode(f, dev))(x, y)
    ref = qml.jacobian(qml.QNode(f, ref_dev, diff_method="backprop"))(x, y)
    for a, b in zip(jac, ref):
        assert qml.math.allclose(a, b, atol=1e-8)


def test_finite_shots_and_other_backends_use_parameter_shift():
    x = pnp.array(0.4, requires_grad=True)

    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.Z(0))

    shots = qml.set_shots(qml.QNode(f, qml.device("maestro.qubit", wires=1)), shots=100)
    assert qml.workflow.get_best_diff_method(shots)(x) == "parameter-shift"

    mps = qml.QNode(f, qml.device("maestro.qubit", wires=1, simulation_type="MatrixProductState"))
    assert qml.workflow.get_best_diff_method(mps)(x) == "parameter-shift"
    assert np.isclose(qml.grad(mps)(x), -np.sin(0.4), atol=1e-8)

    with pytest.raises(qml.exceptions.QuantumFunctionError):
        qml.QNode(f, qml.device("maestro.qubit", wires=1, simulation_type="MatrixProductState"),
                  diff_method="adjoint")(x)


def test_unsupported_measurements_fall_back():
    dev = qml.device("maestro.qubit", wires=2)

    @qml.qnode(dev)
    def probs(x):
        qml.RX(x, wires=0)
        return qml.probs(wires=[0])

    @qml.qnode(dev)
    def mcm(x):
        qml.RX(x, wires=0)
        m = qml.measure(0)
        qml.cond(m, qml.X)(1)
        return qml.expval(qml.Z(1))

    x = pnp.array(0.3, requires_grad=True)
    assert qml.workflow.get_best_diff_method(probs)(x) == "parameter-shift"
    assert qml.math.allclose(qml.jacobian(probs)(x), [-np.sin(0.3) / 2, np.sin(0.3) / 2])
    assert qml.workflow.get_best_diff_method(mcm)(x) == "parameter-shift"


def test_adjoint_of_u3_and_rot_forward():
    """Regression: Adjoint(U3) used to negate all three angles."""

    def f():
        qml.Hadamard(0)
        qml.adjoint(qml.U3(0.3, 0.9, -0.4, wires=0))
        qml.Rot(0.2, -0.5, 1.1, wires=0)
        return qml.expval(qml.X(0)), qml.expval(qml.Y(0))

    res = qml.QNode(f, qml.device("maestro.qubit", wires=1))()
    ref = qml.QNode(f, qml.device("default.qubit", wires=1))()
    assert qml.math.allclose(res, ref, atol=1e-10)


def test_non_diagonal_expval_next_to_state():
    """Regression: the statevector fallback ignored diagonalizing gates."""

    def f():
        qml.Hadamard(0)
        qml.RX(0.3, wires=0)
        qml.RZ(0.7, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.X(0) @ qml.Y(1)), qml.var(qml.Y(0)), qml.probs(wires=[1]), qml.state()

    res = qml.QNode(f, qml.device("maestro.qubit", wires=2))()
    ref = qml.QNode(f, qml.device("default.qubit", wires=2))()
    for a, b in zip(res, ref):
        assert qml.math.allclose(a, b, atol=1e-10)


def test_invalid_adjoint_backend():
    with pytest.raises(ValueError, match="adjoint_backend"):
        qml.device("maestro.qubit", wires=1, adjoint_backend="nope")
