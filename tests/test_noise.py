"""Noise channels: PennyLane's channel operations on the Maestro device.

``default.mixed`` is the reference. A density matrix, an MPO and the Pauli
propagator apply a channel exactly, so analytic results must agree to rounding;
the statevector and the MPS sample one branch of it per shot, so their shots
agree within sampling error.
"""

import maestro
import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp
from pennylane.exceptions import DeviceError

from pennylane_maestro.converter import HAS_CHANNELS, channel_to_maestro

pytestmark = pytest.mark.skipif(
    not HAS_CHANNELS, reason="the installed qoro-maestro has no QuantumChannel"
)

EXACT = ["DensityMatrix", "MatrixProductOperator", "PauliPropagator"]
SAMPLED = ["Statevector", "MatrixProductState"]

# X on the first wire together with Z on the second, with probability 0.2:
# a two-qubit channel that is not symmetric in its wires.
TWO_QUBIT_KRAUS = [
    np.sqrt(0.8) * np.eye(4),
    np.sqrt(0.2) * np.kron(np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]])),
]

CHANNELS = [
    lambda: qml.AmplitudeDamping(0.3, wires=0),
    lambda: qml.GeneralizedAmplitudeDamping(0.25, 0.7, wires=1),
    lambda: qml.PhaseDamping(0.2, wires=2),
    lambda: qml.DepolarizingChannel(0.1, wires=1),
    lambda: qml.BitFlip(0.05, wires=0),
    lambda: qml.PhaseFlip(0.07, wires=2),
    lambda: qml.PauliError("XZ", 0.1, wires=[0, 2]),
    lambda: qml.ResetError(0.1, 0.05, wires=1),
    lambda: qml.ThermalRelaxationError(0.1, 50.0, 70.0, 12.0, wires=2),
    lambda: qml.ThermalRelaxationError(0.2, 70.0, 50.0, 12.0, wires=0),
    lambda: qml.QubitChannel(TWO_QUBIT_KRAUS, wires=[0, 2]),
    lambda: qml.QubitChannel(TWO_QUBIT_KRAUS, wires=[2, 1]),
]


def entangle():
    qml.Hadamard(0)
    qml.RY(0.7, wires=1)
    qml.CNOT([0, 1])
    qml.RX(0.4, wires=2)
    qml.CNOT([1, 2])


def noisy():
    """A three-wire circuit through every channel of ``CHANNELS``."""
    entangle()
    for channel in CHANNELS:
        channel()
    qml.RZ(0.3, wires=0)
    qml.CNOT([2, 0])


def reference(circuit, wires=3):
    return qml.QNode(circuit, qml.device("default.mixed", wires=wires))()


@pytest.mark.parametrize("channel", CHANNELS)
@pytest.mark.parametrize("method", EXACT)
def test_each_channel_is_exact_on_mixed_state_methods(channel, method):
    """One channel at a time, so that a wrong one is named by the failure."""

    def circuit():
        entangle()
        channel()
        return [qml.expval(o) for o in (qml.Z(0), qml.Z(1), qml.Z(2), qml.X(0) @ qml.X(1), qml.Y(1) @ qml.Z(2), qml.X(0) @ qml.Y(2))]

    result = qml.QNode(circuit, qml.device("maestro.qubit", wires=3, simulation_type=method))()
    assert np.allclose(result, reference(circuit), atol=1e-10)


@pytest.mark.parametrize("method", EXACT)
def test_analytic_measurements_match_default_mixed(method):
    """Pauli and Hamiltonian expectation values, variances, a Hermitian
    observable and probabilities, from one noisy circuit."""
    hermitian = np.array([[1, 1j], [-1j, -1]])

    def circuit():
        noisy()
        return (
            qml.expval(qml.Z(0)),
            qml.expval(qml.X(0) @ qml.X(1)),
            qml.expval(0.5 * qml.X(0) + 0.3 * qml.Z(0) @ qml.Z(2)),
            qml.var(qml.Z(1)),
            qml.var(0.5 * qml.X(0) + 0.3 * qml.Z(1)),
            qml.expval(qml.Hermitian(hermitian, wires=2)),
            qml.probs(wires=[0, 2]),
            qml.probs(wires=[2, 0]),
            qml.probs(),
        )

    result = qml.QNode(circuit, qml.device("maestro.qubit", wires=3, simulation_type=method))()
    for ours, theirs in zip(result, reference(circuit)):
        assert np.allclose(ours, theirs, atol=1e-10)


@pytest.mark.parametrize("method", EXACT)
def test_measurements_without_a_statevector_work_without_noise(method):
    """``qml.probs`` and ``qml.var`` on a mixed-state method, which has no
    statevector to read them from."""

    def circuit():
        entangle()
        return qml.probs(wires=[0, 1]), qml.var(qml.X(2)), qml.expval(qml.Z(0) @ qml.Z(1))

    result = qml.QNode(circuit, qml.device("maestro.qubit", wires=3, simulation_type=method))()
    expected = qml.QNode(circuit, qml.device("default.qubit", wires=3))()
    for ours, theirs in zip(result, expected):
        assert np.allclose(ours, theirs, atol=1e-10)


@pytest.mark.parametrize("method", SAMPLED + ["DensityMatrix", "PauliPropagator"])
def test_shots_agree_with_default_mixed(method):
    """Each shot draws its own branch of every channel on the sampling
    methods. Expectation values there average one trajectory per shot."""
    shots = 20000

    def probabilities():
        noisy()
        return qml.probs(wires=[0, 1, 2])

    def values():
        noisy()
        return qml.expval(qml.Z(0)), qml.expval(qml.X(0) @ qml.X(1)), qml.expval(qml.Y(1) @ qml.Z(2))

    device = qml.device("maestro.qubit", wires=3, simulation_type=method, seed=11)
    # Five standard errors of a frequency, and of a +-1 average.
    sampled = qml.set_shots(qml.QNode(probabilities, device), shots)()
    assert np.max(np.abs(sampled - reference(probabilities))) < 5 * np.sqrt(0.25 / shots)
    estimated = qml.set_shots(qml.QNode(values, device), shots)()
    assert np.max(np.abs(np.asarray(estimated) - np.asarray(reference(values)))) < 5 / np.sqrt(shots)


@pytest.mark.parametrize("method", SAMPLED)
def test_sampled_noise_has_no_analytic_result(method):
    """A statevector holds one branch of a channel, not the channel."""

    def circuit():
        noisy()
        return qml.expval(qml.Z(0))

    with pytest.raises(DeviceError, match="Set shots"):
        qml.QNode(circuit, qml.device("maestro.qubit", wires=3, simulation_type=method))()


@pytest.mark.parametrize("method", ["Stabilizer", "TensorNetwork"])
@pytest.mark.parametrize("shots", [None, 100])
def test_methods_that_hold_no_channels_refuse_them(method, shots):
    def circuit():
        qml.Hadamard(0)
        qml.BitFlip(0.1, wires=0)
        return qml.expval(qml.Z(0))

    device = qml.device("maestro.qubit", wires=1, simulation_type=method)
    with pytest.raises(DeviceError, match="takes no noise channels"):
        qml.set_shots(qml.QNode(circuit, device), shots)()


def test_channels_on_more_than_two_wires_are_refused():
    def circuit():
        qml.PauliError("XYZ", 0.1, wires=[0, 1, 2])
        return qml.expval(qml.Z(0))

    with pytest.raises(DeviceError, match="one or two"):
        qml.QNode(circuit, qml.device("maestro.qubit", wires=3, simulation_type="DensityMatrix"))()


def test_a_mixed_state_has_no_statevector():
    def circuit():
        qml.Hadamard(0)
        qml.AmplitudeDamping(0.3, wires=0)
        return qml.state()

    with pytest.raises(DeviceError, match="no statevector"):
        qml.QNode(circuit, qml.device("maestro.qubit", wires=1, simulation_type="DensityMatrix"))()


@pytest.mark.parametrize(
    "op",
    [
        qml.AmplitudeDamping(0.3, wires=0),
        qml.GeneralizedAmplitudeDamping(0.25, 0.7, wires=0),
        qml.PhaseDamping(0.2, wires=0),
        qml.DepolarizingChannel(0.1, wires=0),
        qml.ThermalRelaxationError(0.1, 50.0, 70.0, 12.0, wires=0),
        qml.ThermalRelaxationError(0.2, 70.0, 50.0, 12.0, wires=0),
    ],
)
def test_structured_channels_are_the_map_of_the_kraus_operators(op):
    """Relaxation goes to Maestro in its own form, which a sampling method
    draws more cheaply than a list of Kraus operators. It must be the same
    map, whatever convention PennyLane's parameters follow."""
    kraus = maestro.QuantumChannel.from_kraus([np.asarray(k, dtype=complex) for k in op.kraus_matrices()])
    channel = channel_to_maestro(op)
    assert np.allclose(channel.superoperator(), kraus.superoperator(), atol=1e-12)
    # The structure is kept: relaxation stays relaxation, a Pauli channel a Pauli channel.
    kinds = [factor["kind"] for factor in channel.factors]
    assert kinds == (["pauli"] if op.name in ("PhaseDamping", "DepolarizingChannel") else ["thermal"])


def test_equal_channels_share_one_maestro_channel():
    """A circuit shares the channel object, so it is built once per distinct channel."""
    first = channel_to_maestro(qml.AmplitudeDamping(0.123, wires=0))
    assert channel_to_maestro(qml.AmplitudeDamping(0.123, wires=5)) is first
    assert channel_to_maestro(qml.AmplitudeDamping(0.124, wires=0)) is not first


@pytest.mark.parametrize("method", EXACT)
def test_a_pennylane_noise_model_and_its_gradient(method):
    """``qml.add_noise`` inserts channels after the gates it matches. The
    gradient uses parameter-shift: the adjoint method needs a unitary circuit."""
    model = qml.NoiseModel(
        {
            qml.noise.op_eq(qml.CNOT): qml.noise.partial_wires(qml.DepolarizingChannel, 0.05),
            qml.noise.op_in(["RX", "RY"]): qml.noise.partial_wires(qml.AmplitudeDamping, 0.02),
        }
    )

    def ideal(x):
        qml.RX(x, wires=0)
        qml.RY(0.3, wires=1)
        qml.CNOT([0, 1])
        return qml.expval(qml.Z(0) @ qml.Z(1))

    ours = qml.add_noise(qml.QNode(ideal, qml.device("maestro.qubit", wires=2, simulation_type=method)), model)
    theirs = qml.add_noise(qml.QNode(ideal, qml.device("default.mixed", wires=2)), model)
    x = pnp.array(0.6, requires_grad=True)
    assert np.isclose(ours(x), theirs(x), atol=1e-10)
    assert np.isclose(qml.grad(ours)(x), qml.grad(theirs)(x), atol=1e-8)
    # The noise does something.
    clean = qml.QNode(ideal, qml.device("default.qubit", wires=2))
    assert abs(ours(x) - clean(x)) > 1e-3


def test_a_noisy_circuit_does_not_take_the_adjoint_path():
    """On the CPU statevector the device differentiates unitary circuits with
    the adjoint method; with a channel and shots it must use parameter-shift."""
    device = qml.device("maestro.qubit", wires=1, seed=5)

    @qml.set_shots(40000)
    @qml.qnode(device)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.BitFlip(0.1, wires=0)
        return qml.expval(qml.Z(0))

    x = pnp.array(0.6, requires_grad=True)
    # <Z> = (1 - 2 * 0.1) cos(x)
    assert abs(circuit(x) - 0.8 * np.cos(0.6)) < 0.03
    assert abs(qml.grad(circuit)(x) + 0.8 * np.sin(0.6)) < 0.05


@pytest.mark.parametrize("method", SAMPLED + ["DensityMatrix"])
def test_noise_with_mid_circuit_measurements(method):
    """A measured and reset qubit, then noise on it: each shot is a trajectory."""
    shots = 20000

    def circuit():
        qml.Hadamard(0)
        qml.measure(0, reset=True)
        qml.PauliX(0)
        qml.AmplitudeDamping(0.3, wires=0)
        return qml.probs(wires=0)

    device = qml.device("maestro.qubit", wires=1, simulation_type=method, seed=3)
    result = qml.set_shots(qml.QNode(circuit, device), shots)()
    assert np.max(np.abs(result - np.array([0.3, 0.7]))) < 5 * np.sqrt(0.25 / shots)


def test_an_older_maestro_says_it_has_no_channels(monkeypatch):
    """A qoro-maestro without ``QuantumChannel`` (0.3.14 and earlier) names the
    reason instead of failing somewhere in the conversion."""
    from pennylane_maestro import converter

    monkeypatch.setattr(converter, "HAS_CHANNELS", False)

    def circuit():
        qml.Hadamard(0)
        qml.BitFlip(0.1, wires=0)
        return qml.expval(qml.Z(0))

    with pytest.raises(DeviceError, match="has no QuantumChannel"):
        qml.QNode(circuit, qml.device("maestro.qubit", wires=1, simulation_type="DensityMatrix"))()


def test_noise_does_not_go_through_a_divi_backend():
    """The cloud backends take gates."""
    pytest.importorskip("divi.backends", reason="Divi backends need qoro-divi")
    from pennylane_maestro.maestro_device import _divi_stopping_condition

    with pytest.raises(DeviceError, match="local Maestro engine"):
        _divi_stopping_condition(qml.BitFlip(0.1, wires=0))
