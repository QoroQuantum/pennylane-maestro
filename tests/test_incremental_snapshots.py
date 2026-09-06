"""Tests for persistent incremental evolution and Trotter snapshot acceleration."""

import numpy as np
import pennylane as qml
import pytest

import pennylane_maestro
from pennylane_maestro.maestro_device import (
    _detect_trotter_snapshot_pattern,
    _ops_equal,
    _blocks_equal,
)


class TestTrotterSnapshotPatternDetection:
    """Test pattern detection logic for Trotter circuits with snapshots."""

    def test_detect_uniform_trotter_pattern(self):
        """A simple repeated Trotter circuit with intermediate snapshots matches."""
        num_wires = 2
        ops = [
            qml.PauliX(0),  # init
            qml.IsingXX(0.1, wires=[0, 1]),  # step 1
            qml.Snapshot("s1", measurement=qml.expval(qml.PauliZ(0))),
            qml.IsingXX(0.1, wires=[0, 1]),  # step 2
            qml.Snapshot("s2", measurement=qml.expval(qml.PauliZ(0))),
        ]
        measurements = [qml.expval(qml.PauliZ(0))]

        pattern = _detect_trotter_snapshot_pattern(ops, measurements, num_wires)
        assert pattern is not None
        assert len(pattern["init_ops"]) == 1
        assert pattern["init_ops"][0].name == "PauliX"
        assert len(pattern["step_ops"]) == 1
        assert pattern["step_ops"][0].name == "IsingXX"
        assert pattern["steps"] == [1, 2]
        assert pattern["final_step"] == 2

    def test_detect_multi_step_interval_pattern(self):
        """Snapshots occurring every 2 Trotter steps are detected properly."""
        num_wires = 2
        ops = [
            qml.PauliX(0),
            qml.IsingXX(0.1, wires=[0, 1]),
            qml.IsingXX(0.1, wires=[0, 1]),
            qml.Snapshot("s2", measurement=qml.expval(qml.PauliZ(0))),
            qml.IsingXX(0.1, wires=[0, 1]),
            qml.IsingXX(0.1, wires=[0, 1]),
            qml.Snapshot("s4", measurement=qml.expval(qml.PauliZ(0))),
        ]
        measurements = [qml.expval(qml.PauliZ(0))]

        pattern = _detect_trotter_snapshot_pattern(ops, measurements, num_wires)
        assert pattern is not None
        assert pattern["steps"] == [2, 4]
        assert pattern["final_step"] == 4

    def test_non_matching_different_steps(self):
        """If step blocks have different parameters, pattern detection returns None."""
        num_wires = 2
        ops = [
            qml.PauliX(0),
            qml.IsingXX(0.1, wires=[0, 1]),
            qml.Snapshot("s1", measurement=qml.expval(qml.PauliZ(0))),
            qml.IsingXX(0.2, wires=[0, 1]),  # Different angle
            qml.Snapshot("s2", measurement=qml.expval(qml.PauliZ(0))),
        ]
        measurements = [qml.expval(qml.PauliZ(0))]

        pattern = _detect_trotter_snapshot_pattern(ops, measurements, num_wires)
        assert pattern is None

    def test_full_state_snapshot_returns_none(self):
        """Snapshots requesting full state cannot use Pauli incremental_evolve."""
        num_wires = 2
        ops = [
            qml.PauliX(0),
            qml.IsingXX(0.1, wires=[0, 1]),
            qml.Snapshot("state_1"),  # No measurement kwarg
            qml.IsingXX(0.1, wires=[0, 1]),
            qml.Snapshot("state_2"),
        ]
        measurements = [qml.expval(qml.PauliZ(0))]

        pattern = _detect_trotter_snapshot_pattern(ops, measurements, num_wires)
        assert pattern is None


class TestTrotterSnapshotExecution:
    """Test execution of Trotter circuits with snapshots against default.qubit."""

    def test_heisenberg_evolution_matches_default_qubit(self):
        """Trotterized XX + YY + ZZ evolution with snapshots matches default.qubit."""
        N = 4
        J, dt = 0.5, 0.1
        steps = 4

        def trotter_step():
            for i in range(N - 1):
                qml.IsingXX(2 * J * dt, wires=[i, i + 1])
                qml.IsingYY(2 * J * dt, wires=[i, i + 1])
                qml.IsingZZ(2 * J * dt, wires=[i, i + 1])

        # Reference on default.qubit
        dev_ref = qml.device("default.qubit", wires=N)

        @qml.qnode(dev_ref)
        def ref_circuit():
            # Initial Neel state |0101⟩
            for i in range(1, N, 2):
                qml.PauliX(i)
            for s in range(1, steps + 1):
                trotter_step()
                for q in range(N):
                    qml.Snapshot(f"step_{s}_q{q}", measurement=qml.expval(qml.PauliZ(q)))
            return [qml.expval(qml.PauliZ(i)) for i in range(N)]

        ref_results = qml.snapshots(ref_circuit)()

        # Test on maestro.qubit (MPS backend)
        dev_maestro = qml.device(
            "maestro.qubit",
            wires=N,
            simulation_type="MatrixProductState",
            max_bond_dimension=32,
        )

        @qml.qnode(dev_maestro)
        def maestro_circuit():
            for i in range(1, N, 2):
                qml.PauliX(i)
            for s in range(1, steps + 1):
                trotter_step()
                for q in range(N):
                    qml.Snapshot(f"step_{s}_q{q}", measurement=qml.expval(qml.PauliZ(q)))
            return [qml.expval(qml.PauliZ(i)) for i in range(N)]

        maestro_results = qml.snapshots(maestro_circuit)()

        for s in range(1, steps + 1):
            for q in range(N):
                tag = f"step_{s}_q{q}"
                np.testing.assert_allclose(
                    maestro_results[tag],
                    ref_results[tag],
                    atol=1e-5,
                    err_msg=f"Mismatch at {tag}",
                )

        np.testing.assert_allclose(
            maestro_results["execution_results"],
            ref_results["execution_results"],
            atol=1e-5,
        )

    def test_snapshots_with_hamiltonian_observable(self):
        """Snapshots measuring a Hamiltonian (LinearCombination) work correctly."""
        N = 3
        dev = qml.device(
            "maestro.qubit",
            wires=N,
            simulation_type="MatrixProductState",
            max_bond_dimension=32,
        )

        H = 0.5 * qml.PauliZ(0) + 0.3 * qml.PauliX(1) @ qml.PauliX(2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.Hadamard(1)
            for s in range(1, 4):
                qml.CNOT(wires=[0, 1])
                qml.RZ(0.2, wires=1)
                qml.Snapshot(f"step_{s}", measurement=qml.expval(H))
            return qml.expval(H)

        dev_ref = qml.device("default.qubit", wires=N)

        @qml.qnode(dev_ref)
        def circuit_ref():
            qml.Hadamard(0)
            qml.Hadamard(1)
            for s in range(1, 4):
                qml.CNOT(wires=[0, 1])
                qml.RZ(0.2, wires=1)
                qml.Snapshot(f"step_{s}", measurement=qml.expval(H))
            return qml.expval(H)

        res = qml.snapshots(circuit)()
        res_ref = qml.snapshots(circuit_ref)()

        for s in range(1, 4):
            np.testing.assert_allclose(res[f"step_{s}"], res_ref[f"step_{s}"], atol=1e-5)
        np.testing.assert_allclose(res["execution_results"], res_ref["execution_results"], atol=1e-5)

    def test_multi_observable_snapshot_list(self):
        """Snapshot with measurement as a list of observables."""
        N = 2
        dev = qml.device("maestro.qubit", wires=N, simulation_type="MatrixProductState")

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            for s in range(1, 3):
                qml.RX(0.2, wires=0)
                qml.RY(0.3, wires=1)
                qml.Snapshot(
                    f"step_{s}_Z0",
                    measurement=qml.expval(qml.PauliZ(0)),
                )
                qml.Snapshot(
                    f"step_{s}_Y1",
                    measurement=qml.expval(qml.PauliY(1)),
                )
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        dev_ref = qml.device("default.qubit", wires=N)

        @qml.qnode(dev_ref)
        def circuit_ref():
            qml.PauliX(0)
            for s in range(1, 3):
                qml.RX(0.2, wires=0)
                qml.RY(0.3, wires=1)
                qml.Snapshot(
                    f"step_{s}_Z0",
                    measurement=qml.expval(qml.PauliZ(0)),
                )
                qml.Snapshot(
                    f"step_{s}_Y1",
                    measurement=qml.expval(qml.PauliY(1)),
                )
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        res = qml.snapshots(circuit)()
        res_ref = qml.snapshots(circuit_ref)()

        for s in range(1, 3):
            np.testing.assert_allclose(res[f"step_{s}_Z0"], res_ref[f"step_{s}_Z0"], atol=1e-5)
            np.testing.assert_allclose(res[f"step_{s}_Y1"], res_ref[f"step_{s}_Y1"], atol=1e-5)

    def test_projector_observable_snapshots(self):
        """Test projector Hamiltonian observable like electronic population in vibronic systems."""
        N_EL = 3
        def el_pop_obs(state_idx):
            coeffs, ops = [], []
            for mask in range(2**N_EL):
                c = 1.0 / (2**N_EL)
                paulis = []
                for w in range(N_EL):
                    if (mask >> w) & 1:
                        bit_pos = N_EL - 1 - w
                        sign = -1 if ((state_idx >> bit_pos) & 1) else 1
                        c *= sign
                        paulis.append(qml.PauliZ(w))
                if not paulis:
                    ops.append(qml.Identity(0))
                elif len(paulis) == 1:
                    ops.append(paulis[0])
                else:
                    ops.append(qml.prod(*paulis))
                coeffs.append(c)
            return qml.Hamiltonian(coeffs, ops)

        P0 = el_pop_obs(0)
        P1 = el_pop_obs(1)

        dev_maestro = qml.device(
            "maestro.qubit", wires=N_EL, simulation_type="MatrixProductState"
        )
        dev_ref = qml.device("default.qubit", wires=N_EL)

        def make_circuit(dev):
            @qml.qnode(dev)
            def circuit():
                qml.PauliX(0)
                for s in range(1, 4):
                    qml.RY(0.2, wires=0)
                    qml.CNOT(wires=[0, 1])
                    qml.Snapshot(f"step_{s}_P0", measurement=qml.expval(P0))
                    qml.Snapshot(f"step_{s}_P1", measurement=qml.expval(P1))
                return [qml.expval(P0), qml.expval(P1)]
            return circuit

        res_m = qml.snapshots(make_circuit(dev_maestro))()
        res_r = qml.snapshots(make_circuit(dev_ref))()

        for s in range(1, 4):
            np.testing.assert_allclose(
                res_m[f"step_{s}_P0"], res_r[f"step_{s}_P0"], atol=1e-5
            )
            np.testing.assert_allclose(
                res_m[f"step_{s}_P1"], res_r[f"step_{s}_P1"], atol=1e-5
            )
        np.testing.assert_allclose(
            res_m["execution_results"], res_r["execution_results"], atol=1e-5
        )


class TestDirectIncrementalEvolveAPI:
    """Test device.incremental_evolve direct method and top-level function."""

    def test_direct_api_with_callables(self):
        """Direct incremental_evolve using python callables for init and step."""
        dev = qml.device(
            "maestro.qubit",
            wires=3,
            simulation_type="MatrixProductState",
            max_bond_dimension=16,
        )

        def init():
            qml.PauliX(0)

        def trotter_step():
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.1, wires=1)
            qml.CNOT(wires=[0, 1])

        observables = [
            qml.PauliZ(0),
            qml.PauliZ(1),
            0.5 * qml.PauliZ(0) + 0.5 * qml.PauliZ(1),
        ]
        measure_at_steps = [1, 2, 4]

        res = dev.incremental_evolve(
            init=init,
            trotter_step=trotter_step,
            measure_at_steps=measure_at_steps,
            observables=observables,
        )

        assert "expectation_values" in res
        assert "steps" in res
        assert res["steps"] == [1, 2, 4]
        assert res["expectation_values"].shape == (3, 3)

        # Verify linear combination consistency: O2 = 0.5*O0 + 0.5*O1
        for row in res["expectation_values"]:
            np.testing.assert_allclose(row[2], 0.5 * row[0] + 0.5 * row[1], atol=1e-7)

    def test_top_level_wrapper(self):
        """Top-level pennylane_maestro.incremental_evolve convenience wrapper."""
        dev = qml.device("maestro.qubit", wires=2, simulation_type="MatrixProductState")

        def init():
            qml.Hadamard(0)

        def step():
            qml.CNOT(wires=[0, 1])

        res = pennylane_maestro.incremental_evolve(
            dev,
            init=init,
            trotter_step=step,
            measure_at_steps=[1, 2],
            observables=[qml.PauliZ(0), "Z1"],
        )

        assert res["expectation_values"].shape == (2, 2)
        assert res["steps"] == [1, 2]
