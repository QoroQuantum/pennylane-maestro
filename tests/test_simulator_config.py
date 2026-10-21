"""Tests for SimulatorConfig integration (qoro-maestro ≥ 0.3.7).

Verifies that the device correctly builds a ``maestro.SimulatorConfig``
from user-provided settings and passes it through every execution path.
"""

import pytest
import numpy as np
import pennylane as qml

import maestro
from maestro import SimulatorConfig
from pennylane_maestro.maestro_device import (
    MaestroQubitDevice,
    _SUPPORTS_OBSERVABLE_CLUSTERING,
)


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
        assert cfg.precision is None

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

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_precision_config(self, precision):
        """precision propagates to the config."""
        dev = qml.device("maestro.qubit", wires=1, precision=precision)
        assert dev._build_config().precision == precision

    @pytest.mark.parametrize(
        "use_double_precision, expected", [(True, "double"), (False, "single")]
    )
    def test_legacy_double_precision_config(self, use_double_precision, expected):
        """Legacy use_double_precision maps onto precision."""
        dev = qml.device(
            "maestro.qubit", wires=1, use_double_precision=use_double_precision
        )
        assert dev._build_config().precision == expected

    def test_precision_and_legacy_flag_agree(self):
        """Consistent precision and use_double_precision are accepted."""
        dev = qml.device(
            "maestro.qubit", wires=1, precision="double", use_double_precision=True
        )
        assert dev._build_config().precision == "double"

    def test_precision_conflict_raises(self):
        """Conflicting precision and use_double_precision raise."""
        with pytest.raises(ValueError, match="Conflicting"):
            qml.device(
                "maestro.qubit", wires=1, precision="single", use_double_precision=True
            )

    def test_invalid_precision_raises(self):
        """Unknown precision strings raise."""
        with pytest.raises(ValueError, match="Unknown precision"):
            qml.device("maestro.qubit", wires=1, precision="half")

    def test_optional_fields_unset(self):
        """When max_bond_dimension / SVD threshold are None, config uses defaults."""
        dev = qml.device("maestro.qubit", wires=1)
        cfg = dev._build_config()
        # The default SimulatorConfig should have None for these
        default = SimulatorConfig()
        assert cfg.max_bond_dimension == default.max_bond_dimension
        assert cfg.singular_value_threshold == default.singular_value_threshold

    def test_disable_optimized_swapping_config(self):
        """disable_optimized_swapping flag propagates."""
        dev_default = qml.device("maestro.qubit", wires=1)
        assert dev_default._build_config().disable_optimized_swapping is False

        dev_disabled = qml.device(
            "maestro.qubit", wires=1, disable_optimized_swapping=True
        )
        assert dev_disabled._build_config().disable_optimized_swapping is True

    def test_pauli_propagator_config(self):
        """Pauli propagator parameters are forwarded to the config."""
        dev = qml.device(
            "maestro.qubit",
            wires=2,
            simulation_type="PauliPropagator",
            pp_coefficient_threshold=1e-4,
            pp_max_pauli_weight=5,
            pp_gates_between_trims=10,
            pp_gates_between_deduplications=3,
        )
        cfg = dev._build_config()
        assert cfg.simulation_type == maestro.SimulationType.PauliPropagator
        assert cfg.pp_coefficient_threshold == pytest.approx(1e-4)
        assert cfg.pp_max_pauli_weight == 5
        assert cfg.pp_gates_between_trims == 10
        assert cfg.pp_gates_between_deduplications == 3

    def test_pauli_propagator_defaults_unset(self):
        """Unset Pauli propagator parameters keep SimulatorConfig defaults."""
        cfg = qml.device("maestro.qubit", wires=1)._build_config()
        default = SimulatorConfig()
        assert cfg.pp_max_pauli_weight == default.pp_max_pauli_weight
        assert cfg.pp_gates_between_trims == default.pp_gates_between_trims
        assert (
            cfg.pp_gates_between_deduplications
            == default.pp_gates_between_deduplications
        )

    def test_pauli_propagator_legacy_names(self):
        """Legacy Pauli propagator names map to the new fields with a warning."""
        with pytest.warns(DeprecationWarning) as record:
            dev = qml.device(
                "maestro.qubit",
                wires=2,
                simulation_type="PauliPropagator",
                pp_pauli_weight_threshold=5,
                pp_steps_between_trims=10,
            )
        messages = [str(w.message) for w in record]
        assert any("pp_pauli_weight_threshold" in m for m in messages)
        assert any("pp_steps_between_trims" in m for m in messages)
        cfg = dev._build_config()
        assert cfg.pp_max_pauli_weight == 5
        assert cfg.pp_gates_between_trims == 10

    def test_pauli_propagator_legacy_conflict_raises(self):
        """Conflicting new and legacy Pauli propagator values raise."""
        with pytest.warns(DeprecationWarning), pytest.raises(
            ValueError, match="Conflicting"
        ):
            qml.device(
                "maestro.qubit",
                wires=1,
                pp_max_pauli_weight=4,
                pp_pauli_weight_threshold=5,
            )

    def test_reduction_defaults(self):
        """Pre-execution reductions are on and gate fusion is backend-default."""
        cfg = qml.device("maestro.qubit", wires=1)._build_config()
        assert cfg.enable_causal_cone_reduction is True
        assert cfg.auto_reduce is True
        assert cfg.gate_fusion is None

    def test_reduction_flags_config(self):
        """Pre-execution reduction flags propagate."""
        dev = qml.device(
            "maestro.qubit",
            wires=1,
            enable_causal_cone_reduction=False,
            auto_reduce=False,
        )
        cfg = dev._build_config()
        assert cfg.enable_causal_cone_reduction is False
        assert cfg.auto_reduce is False

    @pytest.mark.parametrize("gate_fusion", [True, False])
    def test_gate_fusion_config(self, gate_fusion):
        """gate_fusion propagates."""
        dev = qml.device("maestro.qubit", wires=1, gate_fusion=gate_fusion)
        assert dev._build_config().gate_fusion is gate_fusion

    @pytest.mark.skipif(
        not _SUPPORTS_OBSERVABLE_CLUSTERING,
        reason="installed qoro-maestro lacks observable clustering",
    )
    def test_observable_clustering_config(self):
        """Observable clustering settings propagate."""
        cfg = qml.device("maestro.qubit", wires=1)._build_config()
        assert cfg.enable_observable_clustering is True
        assert cfg.max_cluster_qubits is None

        dev = qml.device(
            "maestro.qubit",
            wires=1,
            enable_observable_clustering=False,
            max_cluster_qubits=8,
        )
        cfg = dev._build_config()
        assert cfg.enable_observable_clustering is False
        assert cfg.max_cluster_qubits == 8

    @pytest.mark.skipif(
        _SUPPORTS_OBSERVABLE_CLUSTERING,
        reason="installed qoro-maestro supports observable clustering",
    )
    def test_observable_clustering_unsupported_warns(self):
        """Non-default clustering settings warn when Maestro lacks support."""
        with pytest.warns(UserWarning, match="observable clustering"):
            dev = qml.device("maestro.qubit", wires=1, max_cluster_qubits=8)
        assert not hasattr(dev._build_config(), "max_cluster_qubits")

    def test_seed_config(self):
        """Seed setting propagates to SimulatorConfig."""
        dev = qml.device("maestro.qubit", wires=1, seed=42)
        cfg = dev._build_config()
        assert cfg.seed == 42
        assert dev.seed == 42

    def test_seed_reproducibility(self):
        """Identical seeds produce identical shot measurement counts."""
        dev1 = qml.device("maestro.qubit", wires=2, shots=100, seed=12345)
        dev2 = qml.device("maestro.qubit", wires=2, shots=100, seed=12345)

        @qml.qnode(dev1)
        def circuit1():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.counts()

        @qml.qnode(dev2)
        def circuit2():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.counts()

        assert circuit1() == circuit2()

    def test_new_simulator_and_simulation_types(self):
        """Verify new Maestro 0.3.x simulator and simulation types resolve."""
        dev_aer = qml.device("maestro.qubit", wires=1, simulator_type="QiskitAer")
        assert dev_aer._simulator_type == maestro.SimulatorType.QiskitAer

        dev_dist = qml.device("maestro.qubit", wires=1, simulator_type="DistributedGpu")
        assert dev_dist._simulator_type == maestro.SimulatorType.DistributedGpu

        dev_mpo = qml.device(
            "maestro.qubit", wires=1, simulation_type="MatrixProductOperator"
        )
        assert dev_mpo._simulation_type == maestro.SimulationType.MatrixProductOperator


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

    def test_pauli_propagator_execution(self):
        """Pauli propagator execution with SimulatorConfig."""
        dev = qml.device(
            "maestro.qubit",
            wires=2,
            simulation_type="PauliPropagator",
            pp_coefficient_threshold=1e-5,
        )

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        res = circuit()
        # Initial |00>, X on 0 gives |10>. Z@Z on |10> gives -1.
        assert np.isclose(res, -1.0, atol=1e-6)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"enable_causal_cone_reduction": False, "auto_reduce": False},
            {"gate_fusion": True},
            {"gate_fusion": False},
        ],
    )
    def test_reduction_and_fusion_execution(self, kwargs):
        """Results are unchanged with reductions or gate fusion toggled."""
        dev = qml.device("maestro.qubit", wires=3, **kwargs)
        ref = qml.device("default.qubit", wires=3)

        def circuit():
            qml.RX(0.3, 0)
            qml.CNOT([0, 1])
            qml.RY(0.7, 2)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1) @ qml.PauliX(2))

        res = qml.QNode(circuit, dev)()
        expected = qml.QNode(circuit, ref)()
        assert np.allclose(res, expected, atol=1e-6)
