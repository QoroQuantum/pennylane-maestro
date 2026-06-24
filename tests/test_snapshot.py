"""Tests for qml.Snapshot support in pennylane-maestro."""

import numpy as np
import pennylane as qml
import pytest


class TestSnapshotBasic:
    """Basic Snapshot functionality."""

    def test_device_has_debugger(self):
        """Device should have _debugger attribute for snapshot compatibility."""
        dev = qml.device("maestro.qubit", wires=2)
        assert hasattr(dev, "_debugger")
        assert dev._debugger is None

    def test_snapshot_with_pauli_expval(self):
        """Snapshot with Pauli expval measurement uses estimate() fast path."""
        dev = qml.device("maestro.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            qml.Snapshot("after_x", measurement=qml.expval(qml.PauliZ(0)))
            qml.Hadamard(0)
            qml.Snapshot("after_h", measurement=qml.expval(qml.PauliZ(0)))
            return qml.expval(qml.PauliZ(0))

        results = qml.snapshots(circuit)()

        assert "after_x" in results
        assert "after_h" in results
        assert "execution_results" in results
        np.testing.assert_allclose(results["after_x"], -1.0, atol=1e-7)
        np.testing.assert_allclose(results["after_h"], 0.0, atol=1e-7)
        np.testing.assert_allclose(results["execution_results"], 0.0, atol=1e-7)

    def test_snapshot_without_tag(self):
        """Snapshots without explicit tags get integer indices."""
        dev = qml.device("maestro.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Snapshot(measurement=qml.expval(qml.PauliZ(0)))
            qml.PauliX(0)
            qml.Snapshot(measurement=qml.expval(qml.PauliZ(0)))
            return qml.expval(qml.PauliZ(0))

        results = qml.snapshots(circuit)()
        assert 0 in results
        assert 1 in results
        np.testing.assert_allclose(results[0], 1.0, atol=1e-7)
        np.testing.assert_allclose(results[1], -1.0, atol=1e-7)

    def test_snapshot_state_measurement(self):
        """Snapshot without measurement kwarg captures full state (fallback path)."""
        dev = qml.device("maestro.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Snapshot("initial")
            qml.PauliX(0)
            qml.Snapshot("flipped")
            return qml.expval(qml.PauliZ(0))

        results = qml.snapshots(circuit)()
        # Initial state |0⟩
        np.testing.assert_allclose(np.abs(results["initial"]), [1.0, 0.0], atol=1e-7)
        # After X: |1⟩
        np.testing.assert_allclose(np.abs(results["flipped"]), [0.0, 1.0], atol=1e-7)

    def test_snapshot_multi_qubit(self):
        """Snapshot with multi-qubit Pauli expval."""
        N = 4
        dev = qml.device("maestro.qubit", wires=N)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(2)
            qml.PauliX(3)
            qml.Snapshot("t0", measurement=qml.expval(qml.PauliZ(0)))
            qml.Hadamard(0)
            qml.Snapshot("t1", measurement=qml.expval(qml.PauliZ(0)))
            return qml.expval(qml.PauliZ(0))

        results = qml.snapshots(circuit)()
        np.testing.assert_allclose(results["t0"], 1.0, atol=1e-7)
        np.testing.assert_allclose(results["t1"], 0.0, atol=1e-7)
        np.testing.assert_allclose(results["execution_results"], 0.0, atol=1e-7)


class TestSnapshotMPS:
    """Snapshot with MPS backend — the key use case."""

    def test_snapshot_mps_pauli_expval(self):
        """Pauli expval snapshots work with MPS via estimate() fast path."""
        N = 6
        dev = qml.device(
            "maestro.qubit", wires=N,
            simulation_type="MatrixProductState",
            max_bond_dimension=64,
        )

        @qml.qnode(dev)
        def circuit():
            # Domain wall: |000111⟩
            for i in range(N // 2, N):
                qml.PauliX(wires=i)
            qml.Snapshot("t0", measurement=qml.expval(qml.PauliZ(0)))
            qml.Hadamard(0)
            qml.Snapshot("t1", measurement=qml.expval(qml.PauliZ(0)))
            return qml.expval(qml.PauliZ(0))

        results = qml.snapshots(circuit)()
        np.testing.assert_allclose(results["t0"], 1.0, atol=1e-7)
        np.testing.assert_allclose(results["t1"], 0.0, atol=1e-7)

    def test_incremental_trotter_with_snapshots(self):
        """Verify that snapshots during Trotter evolution match
        the rebuild-from-scratch approach."""
        N = 6
        J, dt_step = 1.0, 0.1
        max_steps = 3

        dev = qml.device(
            "maestro.qubit", wires=N,
            simulation_type="MatrixProductState",
            max_bond_dimension=64,
        )

        def apply_bond(i, j, t):
            qml.IsingXX(2 * J * t, wires=[i, j])
            qml.IsingYY(2 * J * t, wires=[i, j])
            qml.IsingZZ(2 * J * t, wires=[i, j])

        def trotter_step():
            for i in range(0, N - 1, 2):
                apply_bond(i, i + 1, dt_step / 2)
            for i in range(1, N - 1, 2):
                apply_bond(i, i + 1, dt_step)
            for i in range(0, N - 1, 2):
                apply_bond(i, i + 1, dt_step / 2)

        # --- Snapshot approach: single circuit ---
        @qml.qnode(dev)
        def snapshot_circuit():
            for i in range(N // 2, N):
                qml.PauliX(wires=i)
            for step in range(1, max_steps + 1):
                trotter_step()
                for q in range(N):
                    qml.Snapshot(
                        f"t{step}_Z{q}",
                        measurement=qml.expval(qml.PauliZ(q)),
                    )
            return [qml.expval(qml.PauliZ(i)) for i in range(N)]

        snap_results = qml.snapshots(snapshot_circuit)()

        # --- Reference: rebuild from scratch ---
        for step in range(1, max_steps + 1):

            @qml.qnode(dev)
            def ref_circuit(n_steps=step):
                for i in range(N // 2, N):
                    qml.PauliX(wires=i)
                for _ in range(n_steps):
                    trotter_step()
                return [qml.expval(qml.PauliZ(i)) for i in range(N)]

            ref_mag = ref_circuit()
            for q in range(N):
                tag = f"t{step}_Z{q}"
                np.testing.assert_allclose(
                    snap_results[tag], ref_mag[q], atol=1e-6,
                    err_msg=f"Mismatch at step={step}, qubit={q}",
                )


class TestSnapshotEdgeCases:
    """Edge cases and regression tests."""

    def test_no_snapshots_unchanged(self):
        """Circuits without snapshots still work normally."""
        dev = qml.device("maestro.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        result = circuit()
        np.testing.assert_allclose(result, -1.0, atol=1e-7)

    def test_snapshot_at_beginning(self):
        """Snapshot before any gates captures initial state."""
        dev = qml.device("maestro.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Snapshot("start", measurement=qml.expval(qml.PauliZ(0)))
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        results = qml.snapshots(circuit)()
        np.testing.assert_allclose(results["start"], 1.0, atol=1e-7)
        np.testing.assert_allclose(results["execution_results"], -1.0, atol=1e-7)

    def test_many_snapshots(self):
        """Many snapshots in one circuit all produce correct results."""
        dev = qml.device("maestro.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            for i in range(10):
                qml.RX(0.1 * (i + 1), wires=0)
                qml.Snapshot(f"step_{i}", measurement=qml.expval(qml.PauliZ(0)))
            return qml.expval(qml.PauliZ(0))

        results = qml.snapshots(circuit)()
        assert len(results) == 11  # 10 snapshots + execution_results
        # Cumulative RX rotation: after step i, total angle = 0.1*(1+2+...+(i+1))
        # = 0.1 * (i+1)*(i+2)/2.  ⟨Z⟩ = cos(angle)
        cumulative = 0.0
        for i in range(10):
            cumulative += 0.1 * (i + 1)
            expected_z = np.cos(cumulative)
            np.testing.assert_allclose(
                results[f"step_{i}"], expected_z, atol=1e-6,
                err_msg=f"Mismatch at step {i}",
            )
