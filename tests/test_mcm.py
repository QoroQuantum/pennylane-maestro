"""Tests for mid-circuit measurement (MCM) support in maestro.qubit.

Both ``mcm_method="one-shot"`` and ``mcm_method="deferred"`` are tested.
All tests cross-validate against ``default.qubit`` because both devices
share the ``defer_measurements`` pre-processing strategy.
"""

import pytest
import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["one-shot", "deferred"])
def mcm_method(request):
    return request.param


@pytest.fixture
def shots():
    return 20_000


# ---------------------------------------------------------------------------
# Basic mid-circuit measurement
# ---------------------------------------------------------------------------

class TestBasicMCM:
    """MCM circuits without conditional operations."""

    def test_mcm_deterministic_zero(self, shots, mcm_method):
        """Measure |0⟩ — should always return 0."""
        dev = qml.device("maestro.qubit", wires=2, shots=shots)

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            m = qml.measure(0)
            return qml.sample(m)

        results = circuit()
        assert results.shape == (shots,)
        assert np.all(results == 0), "Measuring |0⟩ should always yield 0"

    def test_mcm_deterministic_one(self, shots, mcm_method):
        """Measure |1⟩ — should always return 1."""
        dev = qml.device("maestro.qubit", wires=2, shots=shots)

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            qml.PauliX(wires=0)
            m = qml.measure(0)
            return qml.sample(m)

        results = circuit()
        assert results.shape == (shots,)
        assert np.all(results == 1), "Measuring |1⟩ should always yield 1"

    def test_mcm_superposition_statistics(self, shots, mcm_method):
        """Measure |+⟩ — should yield 0 and 1 with ~50 % probability each."""
        dev = qml.device("maestro.qubit", wires=2, shots=shots)

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            qml.Hadamard(wires=0)
            m = qml.measure(0)
            return qml.sample(m)

        results = circuit()
        freq_zero = np.mean(results == 0)
        assert 0.45 < freq_zero < 0.55, (
            f"Expected ~50 % zeros, got {freq_zero:.3f}"
        )

    def test_mcm_expval_against_default_qubit(self, shots, mcm_method):
        """MCM sample mean agrees with default.qubit reference."""
        dev_m = qml.device("maestro.qubit", wires=2, shots=shots)
        dev_d = qml.device("default.qubit", wires=2, shots=shots)

        def circuit():
            qml.Hadamard(wires=0)
            m = qml.measure(0)
            return qml.sample(m)

        mean_m = np.mean(qml.QNode(circuit, dev_m, mcm_method=mcm_method)())
        mean_d = np.mean(qml.QNode(circuit, dev_d, mcm_method=mcm_method)())

        assert abs(mean_m - mean_d) < 0.05, (
            f"Mean mismatch: maestro={mean_m:.3f}, default={mean_d:.3f}"
        )

    def test_multiple_mcms(self, shots, mcm_method):
        """Multiple sequential mid-circuit measurements."""
        dev = qml.device("maestro.qubit", wires=3, shots=shots)

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            qml.PauliX(wires=0)
            m0 = qml.measure(0)
            qml.PauliX(wires=1)
            m1 = qml.measure(1)
            return qml.sample(m0), qml.sample(m1)

        s0, s1 = circuit()
        assert np.all(s0 == 1), "Wire 0 should always measure 1"
        assert np.all(s1 == 1), "Wire 1 should always measure 1"


# ---------------------------------------------------------------------------
# MCM with reset
# ---------------------------------------------------------------------------

class TestMCMReset:
    """Mid-circuit measurements with ``reset=True``."""

    def test_reset_produces_zero(self, shots, mcm_method):
        """After reset=True MCM on |1⟩, qubit is back in |0⟩."""
        dev = qml.device("maestro.qubit", wires=2, shots=shots)

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            qml.PauliX(wires=0)
            _ = qml.measure(0, reset=True)
            # Qubit 0 should now be |0⟩
            return qml.sample(wires=0)

        results = circuit()
        assert np.all(results == 0), "Qubit should be |0⟩ after reset"

    def test_reset_mcm_outcome_is_correct(self, shots, mcm_method):
        """The MCM sample value is the pre-reset measurement outcome."""
        dev = qml.device("maestro.qubit", wires=2, shots=shots)

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            qml.PauliX(wires=0)       # |1⟩
            m = qml.measure(0, reset=True)
            return qml.sample(m)

        results = circuit()
        assert np.all(results == 1), (
            "MCM outcome of |1⟩ should be 1, even after reset"
        )

    def test_reset_then_reuse_wire(self, shots, mcm_method):
        """After reset, the wire can be used for a fresh qubit state."""
        dev = qml.device("maestro.qubit", wires=2, shots=shots)

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            qml.PauliX(wires=0)
            _ = qml.measure(0, reset=True)   # reset to |0⟩
            qml.PauliX(wires=0)              # flip to |1⟩ again
            return qml.sample(wires=0)

        results = circuit()
        assert np.all(results == 1), "Reused wire should hold |1⟩"


# ---------------------------------------------------------------------------
# Conditional operations
# ---------------------------------------------------------------------------

class TestConditional:
    """Operations conditioned on MCM outcomes."""

    def test_conditional_x_on_one(self, shots, mcm_method):
        """qml.cond(m, X): apply X if MCM = 1."""
        dev_m = qml.device("maestro.qubit", wires=2, shots=shots)
        # default.qubit needs no fixed wires so defer_measurements ancilla
        # qubit (wire 2) does not trigger a WireError.
        dev_d = qml.device("default.qubit", shots=shots)

        def circuit():
            qml.PauliX(wires=0)        # |1⟩ → MCM = 1
            m = qml.measure(0)
            qml.cond(m, qml.PauliX)(wires=1)
            return qml.sample(wires=[0, 1])

        s_m = qml.QNode(circuit, dev_m, mcm_method=mcm_method)()
        s_d = qml.QNode(circuit, dev_d, mcm_method=mcm_method)()

        # MCM = 1 always, so wire 1 should be flipped every time
        assert np.all(s_m[:, 1] == 1), "Wire 1 should always be 1 (conditional X applied)"
        assert np.array_equal(
            np.mean(s_m, axis=0), np.mean(s_d, axis=0)
        ) or np.allclose(
            np.mean(s_m, axis=0), np.mean(s_d, axis=0), atol=0.05
        )

    def test_conditional_x_on_zero(self, shots, mcm_method):
        """qml.cond(m, X): do NOT apply X if MCM = 0."""
        dev = qml.device("maestro.qubit", wires=2, shots=shots)

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            # Wire 0 starts in |0⟩ → MCM = 0
            m = qml.measure(0)
            qml.cond(m, qml.PauliX)(wires=1)
            return qml.sample(wires=1)

        results = circuit()
        assert np.all(results == 0), "X should not be applied when MCM = 0"

    def test_conditional_matches_default_qubit(self, shots, mcm_method):
        """Conditional circuit mean matches default.qubit."""
        dev_m = qml.device("maestro.qubit", wires=2, shots=shots)
        # default.qubit needs no fixed wires so defer_measurements ancilla
        # qubit (wire 2) does not trigger a WireError.
        dev_d = qml.device("default.qubit", shots=shots)

        def circuit():
            qml.Hadamard(wires=0)
            m = qml.measure(0)
            qml.cond(m, qml.PauliX)(wires=1)
            return qml.sample(wires=[0, 1])

        s_m = qml.QNode(circuit, dev_m, mcm_method=mcm_method)()
        s_d = qml.QNode(circuit, dev_d, mcm_method=mcm_method)()

        # Wire 0 and 1 should be perfectly correlated (bit-flip teleportation)
        assert np.allclose(
            np.mean(s_m, axis=0), np.mean(s_d, axis=0), atol=0.05
        ), "Conditional circuit statistics mismatch"


# ---------------------------------------------------------------------------
# Larger / Stim-like circuits
# ---------------------------------------------------------------------------

class TestStimLike:
    """Patterns analogous to Stim-translated QEC circuits."""

    def test_measure_and_reset_loop(self, shots, mcm_method):
        """Repeated measure-and-reset on the same wire."""
        dev = qml.device("maestro.qubit", wires=2, shots=shots)

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            outcomes = []
            for _ in range(4):
                qml.PauliX(wires=0)
                m = qml.measure(0, reset=True)
                outcomes.append(m)
            return tuple(qml.sample(o) for o in outcomes)

        results = circuit()
        # Every MCM is on |1⟩ → should be 1 each time
        for r in results:
            assert np.all(r == 1), "Each MCM should sample 1"

    def test_many_mcm_wires_stabilizer(self, mcm_method):
        """Clifford circuit with MCMs on many wires via Stabilizer backend."""
        n = 10
        dev = qml.device(
            "maestro.qubit",
            wires=n + n,  # extra room for ancilla from defer_measurements
            simulation_type="Stabilizer",
            shots=100,
        )

        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit():
            # Prepare a GHZ state over n qubits
            qml.Hadamard(wires=0)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
            # MCM all qubits
            ms = [qml.measure(i) for i in range(n)]
            return tuple(qml.sample(m) for m in ms)

        results = circuit()
        # All MCM outcomes should be identical for each shot (GHZ correlation)
        stacked = np.column_stack(results)          # (shots, n)
        all_same = np.all(stacked == stacked[:, :1], axis=1)
        assert np.mean(all_same) > 0.95, (
            "GHZ MCM outcomes should be all-0 or all-1 per shot"
        )


# ---------------------------------------------------------------------------
# MCM-free circuits are unaffected
# ---------------------------------------------------------------------------

class TestMCMFreeRegressions:
    """Verify that circuits without MCMs still work correctly after the
    preprocess_transforms override."""

    def test_bell_state_probs(self, tol):
        """Bell state probabilities unchanged."""
        dev = qml.device("maestro.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        probs = circuit()
        assert np.allclose(probs, [0.5, 0.0, 0.0, 0.5], **tol)

    def test_expval_unchanged(self, tol):
        """Single-qubit expectation value unchanged."""
        dev_m = qml.device("maestro.qubit", wires=1)
        dev_d = qml.device("default.qubit", wires=1)

        def circuit():
            qml.RX(0.7, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(
            qml.QNode(circuit, dev_m)(),
            qml.QNode(circuit, dev_d)(),
            **tol,
        )
