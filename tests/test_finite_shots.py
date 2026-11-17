"""Finite shots: every measurement reads samples taken in its own basis.

Samples are computational-basis outcomes. An observable that is not diagonal
there needs the circuit rotated into its eigenbasis before it is sampled, and
measurements that need different rotations need separate runs. The reference
is ``default.qubit`` without shots, whose values are exact.
"""

import numpy as np
import pennylane as qml
import pytest

from pennylane_maestro import converter
from pennylane_maestro.maestro_device import _sampling_plan

SHOTS = 20_000
ATOL = 0.04

HERMITIAN_1 = np.array([[1.0, 1.0j], [-1.0j, 0.5]])
HERMITIAN_2 = np.kron(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, -1.0]])) + np.diag([0.3, -0.2, 0.1, 0.4])


def _prepare():
    """A state with no symmetry that would hide a wrong basis."""
    qml.RX(0.7, wires=0)
    qml.RY(1.1, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(0.4, wires=1)
    qml.RY(-0.5, wires=2)
    qml.CNOT(wires=[1, 2])
    qml.RX(0.9, wires=0)


def _sample_wire():
    """Next to it in a tape, the other measurements are read from samples."""
    return qml.sample(wires=[0])


def _run(measurements, shots, **options):
    """The measurements on maestro.qubit with shots, and exactly on default.qubit
    (but for the samples, which have no exact value)."""
    sampled = qml.device("maestro.qubit", wires=3, shots=shots, seed=11, **options)
    exact = qml.device("default.qubit", wires=3)

    def circuit(chosen):
        _prepare()
        return [m() for m in chosen]

    reference = [m for m in measurements if m is not _sample_wire]
    return qml.QNode(circuit, sampled)(measurements), qml.QNode(circuit, exact)(reference)


MEASUREMENTS = {
    "var(X)": lambda: qml.var(qml.X(0)),
    "var(Y)": lambda: qml.var(qml.Y(1)),
    "var(Z)": lambda: qml.var(qml.Z(2)),
    "var(Hadamard)": lambda: qml.var(qml.Hadamard(0)),
    "var(X@Y)": lambda: qml.var(qml.X(0) @ qml.Y(1)),
    "var(X@Z@Y)": lambda: qml.var(qml.X(0) @ qml.Z(1) @ qml.Y(2)),
    "var(X+Z)": lambda: qml.var(qml.X(0) + qml.Z(0)),
    "var(X+X)": lambda: qml.var(qml.X(0) + qml.X(1)),
    "expval(Hermitian)": lambda: qml.expval(qml.Hermitian(HERMITIAN_1, wires=1)),
    "var(Hermitian)": lambda: qml.var(qml.Hermitian(HERMITIAN_1, wires=1)),
    "expval(Hermitian on two wires)": lambda: qml.expval(qml.Hermitian(HERMITIAN_2, wires=[0, 2])),
    "expval(Projector)": lambda: qml.expval(qml.Projector([1], wires=1)),
    "probs": lambda: qml.probs(wires=[0, 1, 2]),
    "probs of some wires": lambda: qml.probs(wires=[2, 0]),
}


@pytest.mark.parametrize("name", list(MEASUREMENTS))
def test_a_measurement_with_shots_agrees_with_the_exact_value(name):
    """Alone in its tape next to a sample of the wires, so that it is read from samples."""
    (value, samples), (expected,) = _run([MEASUREMENTS[name], _sample_wire], SHOTS)
    assert np.asarray(samples).shape[0] == SHOTS
    assert np.allclose(value, expected, atol=ATOL), f"{name}: {value} with shots, {expected} exactly"


def test_measurements_in_different_bases_share_a_tape():
    """Each is right although no single set of samples serves them all."""
    names = ["var(X)", "var(Z)", "var(Y)", "expval(Hermitian)", "var(X@Y)", "probs", "var(X+Z)", "expval(Hermitian on two wires)"]
    values, expected = _run([MEASUREMENTS[name] for name in names], SHOTS)
    for name, value, reference in zip(names, values, expected):
        assert np.allclose(value, reference, atol=ATOL), f"{name}: {value} with shots, {reference} exactly"


def test_a_sum_of_words_that_do_not_commute_is_measured_term_by_term():
    """Next to a sample, an expectation value of a sum is read from samples: each word in its own basis."""
    hamiltonian = 0.5 * qml.X(0) + 1.5 * qml.Z(0) - 0.7 * (qml.Y(1) @ qml.X(2)) + 0.25 * qml.I(0) + 0.4 * (qml.Z(1) @ qml.Z(2))
    (value, _), (expected,) = _run([lambda: qml.expval(hamiltonian), _sample_wire], SHOTS)
    assert np.isclose(value, expected, atol=2 * ATOL)


@pytest.mark.parametrize("observable", [
    lambda: qml.X(0) + qml.Z(0),
    lambda: 0.5 * qml.X(0) + 1.5 * qml.Z(0) - 0.7 * (qml.Y(1) @ qml.X(2)),
    lambda: qml.Hamiltonian([0.3, -1.2, 0.8], [qml.X(0) @ qml.Y(1), qml.Z(2), qml.Y(0)]),
])
def test_expectation_values_of_sums_alone_are_estimated(observable):
    """As for Pauli words, a tape of expectation values of sums of words takes Maestro's estimate, which has no shot noise."""
    (value,), (expected,) = _run([lambda: qml.expval(observable())], 50)
    assert np.isclose(value, expected, atol=1e-9)


def test_an_eigenstate_has_no_variance():
    """|+> is an eigenstate of X: every sample taken in the X basis reads +1."""
    dev = qml.device("maestro.qubit", wires=1, shots=1000, seed=3)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.var(qml.X(0)), qml.sample(qml.X(0)), qml.counts(qml.X(0)), qml.expval(qml.X(0) + qml.Z(0))

    variance, samples, counts, total = circuit()
    assert variance == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.asarray(samples) == 1)
    assert {float(key): int(count) for key, count in counts.items()} == {1.0: 1000}
    # <X> is exactly 1, and <Z> is 0 up to shot noise.
    assert total == pytest.approx(1.0, abs=0.12)


def test_samples_of_an_observable_are_its_eigenvalues():
    dev = qml.device("maestro.qubit", wires=3, shots=SHOTS, seed=11)

    @qml.qnode(dev)
    def circuit():
        _prepare()
        return qml.sample(qml.Y(1)), qml.expval(qml.Y(1))

    samples, mean = circuit()
    samples = np.asarray(samples)
    assert set(np.unique(samples)) <= {-1, 1}
    (_,), (expected,) = _run([lambda: qml.expval(qml.Y(1))], 10)
    assert np.mean(samples) == pytest.approx(expected, abs=ATOL)
    assert mean == pytest.approx(np.mean(samples), abs=1e-12)  # the same samples serve both


def test_a_shot_vector_reads_every_part_in_the_right_basis():
    values, expected = _run([MEASUREMENTS["var(X)"], MEASUREMENTS["var(Y)"]], [SHOTS, SHOTS // 2])
    assert len(values) == 2
    for part in values:
        assert np.allclose(part, expected, atol=ATOL)


@pytest.mark.parametrize("method", ["Statevector", "MatrixProductState", "DensityMatrix"])
def test_other_simulation_types_rotate_too(method):
    values, expected = _run([MEASUREMENTS["var(X)"], MEASUREMENTS["var(Y)"], MEASUREMENTS["expval(Hermitian)"]], SHOTS,
                            simulation_type=method)
    assert np.allclose(values, expected, atol=ATOL)


def test_a_native_mid_circuit_measurement_keeps_the_terminal_bases():
    """With native mid-circuit measurements everything is sampled: a Pauli expectation value too."""
    dev = qml.device("maestro.qubit", wires=2, shots=2000, seed=5)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        outcome = qml.measure(0)
        return qml.expval(qml.X(1)), qml.var(qml.X(1)), qml.expval(qml.X(1) + qml.Z(1)), qml.sample(outcome), qml.expval(qml.Z(0))

    in_x, variance, total, outcomes, collapsed = circuit()
    assert in_x == pytest.approx(1.0, abs=1e-12)
    assert variance == pytest.approx(0.0, abs=1e-12)
    assert total == pytest.approx(1.0, abs=0.1)
    outcomes = np.asarray(outcomes)
    assert outcomes.shape == (2000,) and abs(outcomes.mean() - 0.5) < 0.05
    # The measured wire holds its outcome: <Z> over the shots is that of the outcomes (of another run).
    assert collapsed == pytest.approx(0.0, abs=0.1)


@pytest.mark.skipif(not converter.HAS_CHANNELS, reason="the installed qoro-maestro has no QuantumChannel")
@pytest.mark.parametrize("method", ["Statevector", "DensityMatrix"])
def test_noise_is_read_in_the_right_basis(method):
    """A phase flip of probability p takes <X> of |+> from 1 to 1 - 2p."""
    dev = qml.device("maestro.qubit", wires=1, shots=SHOTS, seed=9, simulation_type=method)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.PhaseFlip(0.2, wires=0)
        return qml.var(qml.X(0)), qml.sample(qml.X(0))

    variance, samples = circuit()
    assert variance == pytest.approx(1 - 0.6 ** 2, abs=ATOL)
    assert np.mean(samples) == pytest.approx(0.6, abs=ATOL)


def test_measurements_that_agree_on_every_wire_share_one_run():
    """One run per basis, not per measurement."""
    def groups(*measurements):
        tape = qml.tape.QuantumScript([qml.Hadamard(0)], list(measurements), shots=10)
        return len(_sampling_plan(tape, 3)[0])

    assert groups(qml.sample(), qml.probs(wires=[0, 1]), qml.var(qml.Z(2)), qml.counts()) == 1
    assert groups(qml.var(qml.X(0)), qml.var(qml.Y(1)), qml.var(qml.X(0) @ qml.Y(1)), qml.sample(qml.Z(2))) == 1
    assert groups(qml.var(qml.X(0)), qml.var(qml.Y(0))) == 2
    assert groups(qml.var(qml.X(0)), qml.sample()) == 2
    # X0 and Z1 Z2 fit one run; Z0 needs another; the identity needs none.
    assert groups(qml.expval(qml.X(0) + qml.Z(0) + qml.Z(1) @ qml.Z(2) + 2.0 * qml.I(0)), qml.var(qml.Z(2))) == 2
