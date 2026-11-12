"""Running maestro.qubit circuits through Divi (``backend=``).

Divi's local ``MaestroSimulator`` stands in for the cloud: it implements the
same ``CircuitRunner`` contract as ``QoroService`` (expectation-value jobs,
counts keyed with c[0] first, ``submit_vjp``). ``TestQoroService`` drives the real
``QoroService`` client offline through a fake HTTP transport.
"""

import base64
import gzip
import itertools
import re
import subprocess
import sys
import warnings
from unittest import mock

import maestro
import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp
from pennylane.exceptions import DeviceError, QuantumFunctionError
from pennylane.tape import QuantumScript

divi = pytest.importorskip("divi", reason="Divi backends need qoro-divi")

from divi.backends import (  # noqa: E402
    JobConfig,
    MaestroConfig,
    MaestroSimulator,
    QoroService,
    QPUSystem,
    SimulatorCluster,
)

from pennylane_maestro import divi_backend  # noqa: E402

H = 0.5 * qml.Z(0) @ qml.X(1) - 0.3 * qml.Y(2) + 0.2 * qml.I(0)


class SpyMaestro(MaestroSimulator):
    """MaestroSimulator recording every job it is sent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jobs = []

    def submit_circuits(self, payloads, **kwargs):
        self.jobs.append(("circuits", list(payloads), kwargs))
        return super().submit_circuits(payloads, **kwargs)

    def submit_vjp(self, payloads, **kwargs):
        self.jobs.append(("vjp", list(payloads), kwargs))
        return super().submit_vjp(payloads, **kwargs)


class NoVJPMaestro(SpyMaestro):
    @property
    def supports_vjp(self):
        return False


class DropRowMaestro(SpyMaestro):
    """Loses the last row of every VJP job."""

    def submit_vjp(self, payloads, **kwargs):
        out = super().submit_vjp(payloads, **kwargs)
        return out.with_results(out.results[:-1])


@pytest.fixture
def runner():
    return SpyMaestro(maestro_config=MaestroConfig(seed=11))


def _device(backend, wires=3, **kwargs):
    return qml.device("maestro.qubit", wires=wires, backend=backend, **kwargs)


def _ansatz(x, wires=3):
    qml.RX(x[0], 0)
    qml.RY(x[1], 1)
    qml.CNOT([0, 1])
    qml.Rot(x[2], x[0], x[1], wires - 1)
    qml.CRZ(x[1], [wires - 1, 0])
    qml.IsingXX(x[2], [1, wires - 1])


X0 = pnp.array([0.3, -0.7, 1.1], requires_grad=True)


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------


class TestSetting:
    def test_local_has_no_backend(self):
        assert qml.device("maestro.qubit", wires=2).backend is None

    def test_runner_is_used_as_is(self, runner):
        assert _device(runner).backend is runner

    def test_unknown_backend_string(self):
        with pytest.raises(ValueError, match="Unknown backend 'cloud'"):
            qml.device("maestro.qubit", wires=2, backend="cloud")

    def test_backend_must_be_a_runner(self):
        with pytest.raises(TypeError, match="CircuitRunner"):
            qml.device("maestro.qubit", wires=2, backend=object())

    def test_default_device_options_send_nothing(self, runner):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _device(runner)

    def test_device_options_do_not_reach_a_runner(self, runner):
        with pytest.warns(UserWarning, match="max_bond_dimension"):
            _device(runner, max_bond_dimension=16)

    def test_explicit_engine_choice_does_not_reach_a_runner(self, runner):
        with pytest.warns(UserWarning, match="simulation_type"):
            _device(runner, simulation_type="Statevector")

    def test_divi_import_keeps_the_excepthook(self):
        code = (
            "import sys; hook = sys.excepthook\n"
            "from pennylane_maestro import divi_backend\n"
            "divi_backend.import_divi()\n"
            "import divi\n"
            "print(sys.excepthook is hook)\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip().splitlines()[-1] == "True"

    def test_incremental_evolve_is_local(self, runner):
        with pytest.raises(DeviceError, match="local Maestro engine"):
            _device(runner).incremental_evolve([], [qml.RX(0.1, 0)], [1], [qml.Z(0)])


# ---------------------------------------------------------------------------
# Analytic results
# ---------------------------------------------------------------------------

ANALYTIC_MEASUREMENTS = {
    "pauli": lambda: [qml.expval(qml.Z(0) @ qml.Y(2))],
    "scaled": lambda: [qml.expval(-2.5 * qml.X(1))],
    "hamiltonian": lambda: [qml.expval(H)],
    "hermitian": lambda: [qml.expval(qml.Hermitian(np.array([[1, 1j], [-1j, 2]]), 1))],
    "projector": lambda: [qml.expval(qml.Projector([1, 0], wires=[0, 2]))],
    "var_pauli": lambda: [qml.var(qml.X(1))],
    "var_hamiltonian": lambda: [qml.var(H)],
    "probs_subset": lambda: [qml.probs(wires=[2, 0])],
    "probs_all": lambda: [qml.probs()],
    "probs_of_pauli": lambda: [qml.probs(op=qml.X(0)), qml.probs(op=qml.Z(0) @ qml.Y(2))],
    "probs_of_hermitian": lambda: [qml.probs(op=qml.Hermitian(np.array([[1, 1j], [-1j, 2]]), 1))],
    "var_hermitian": lambda: [qml.var(qml.Hermitian(np.array([[1, 1j], [-1j, 2]]), 1))],
    "hermitian_reversed_wires": lambda: [
        qml.expval(qml.Hermitian(np.diag([1.0, 2.0, -1.0, 0.5]) + 0.3 * np.eye(4)[::-1], wires=[2, 0]))
    ],
    "mixed": lambda: [qml.expval(H), qml.var(qml.Y(0)), qml.probs(wires=1), qml.expval(qml.Z(2))],
}


class TestAnalytic:
    @pytest.mark.parametrize("name", list(ANALYTIC_MEASUREMENTS))
    def test_matches_default_qubit(self, runner, name, tol):
        def circuit(x):
            _ansatz(x)
            return [m for m in ANALYTIC_MEASUREMENTS[name]()]

        got = qml.QNode(circuit, _device(runner))(X0)
        want = qml.QNode(circuit, qml.device("default.qubit", wires=3))(X0)
        for g, w in zip(got, want):
            assert np.allclose(g, w, **tol)

    def test_single_measurement_is_bare(self, runner):
        @qml.qnode(_device(runner))
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.X(0))

        result = circuit()
        assert isinstance(result, np.float64) and np.isclose(result, 1.0)

    def test_batch_is_one_expectation_job(self, runner):
        tapes = [
            QuantumScript([qml.RX(0.1 * k, 0), qml.CNOT([0, 1])], [qml.expval(obs)])
            for k, obs in enumerate([qml.Z(0), qml.Z(0), qml.X(1) @ qml.Z(0), H])
        ]
        results = qml.execute(tapes, _device(runner))
        expected = qml.execute(tapes, qml.device("default.qubit", wires=3))
        assert np.allclose(results, expected)
        assert len(runner.jobs) == 1
        kind, payloads, kwargs = runner.jobs[0]
        assert kind == "circuits" and len(payloads) == 4
        # Each run of tapes with the same terms shares one observable group.
        assert kwargs["circuit_ham_map"] == [[0, 2], [2, 3], [3, 4]]

    def test_identity_observables_need_no_job(self, runner):
        @qml.qnode(_device(runner))
        def circuit():
            qml.RX(0.3, 0)
            return qml.expval(2.0 * qml.I(0))

        assert np.isclose(circuit(), 2.0)
        assert runner.jobs == []

    @pytest.mark.parametrize("wires", [["a", "b", 7], [2, 0, 1]])
    def test_wire_labels(self, runner, wires, tol):
        def circuit():
            qml.X(wires[2])
            qml.RY(0.3, wires[1])
            qml.CNOT([wires[1], wires[0]])
            return qml.probs(wires=[wires[2], wires[0]]), qml.expval(qml.Z(wires[1]) @ qml.Z(wires[2]))

        got = qml.QNode(circuit, qml.device("maestro.qubit", wires=wires, backend=runner))()
        want = qml.QNode(circuit, qml.device("default.qubit", wires=wires))()
        for g, w in zip(got, want):
            assert np.allclose(g, w, **tol)

    def test_unsorted_integer_wires(self, runner, tol):
        # Integer wires first used out of order stay on their own qubits.
        tape = QuantumScript(
            [qml.X(1), qml.RX(0.4, 0)], [qml.expval(qml.Z(0)), qml.probs(wires=[1])]
        )
        (got,) = qml.execute([tape], _device(runner, wires=2))
        (want,) = qml.execute([tape], qml.device("default.qubit", wires=2))
        for g, w in zip(got, want):
            assert np.allclose(g, w, **tol)

    def test_mid_circuit_measurements(self, runner, tol):
        def circuit(x):
            qml.RX(x, 0)
            m = qml.measure(0)
            qml.cond(m, qml.X)(1)
            return qml.expval(m), qml.probs(op=m), qml.var(m), qml.probs(wires=1)

        got = qml.QNode(circuit, _device(runner))(0.4)
        want = qml.QNode(circuit, qml.device("default.qubit", wires=4))(0.4)
        for g, w in zip(got, want):
            assert np.allclose(g, w, **tol)

    def test_broadcasting(self, runner, tol):
        @qml.qnode(_device(runner))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0)), qml.probs(wires=0)

        x = np.array([0.1, 0.7, 2.0])
        expval, probs = circuit(x)
        assert np.allclose(expval, np.cos(x), **tol)
        assert np.allclose(probs[:, 0], np.cos(x / 2) ** 2, **tol)

    def test_snapshots(self, runner, tol):
        def circuit(x):
            qml.RX(x, 0)
            qml.Snapshot("first", measurement=qml.expval(qml.Z(0)))
            qml.RX(x, 0)
            qml.Snapshot(measurement=qml.probs(wires=0))
            return qml.expval(qml.Z(0))

        got = qml.snapshots(qml.QNode(circuit, _device(runner)))(0.4)
        want = qml.snapshots(qml.QNode(circuit, qml.device("default.qubit", wires=3)))(0.4)
        assert got.keys() == want.keys()
        for key in want:
            assert np.allclose(got[key], want[key], **tol)
        # Without qml.snapshots the snapshots are ignored.
        assert np.isclose(qml.QNode(circuit, _device(runner))(0.4), np.cos(0.8))

    def test_analytic_snapshot_in_a_sampled_circuit(self, runner, tol):
        @qml.qnode(_device(runner), shots=50)
        def circuit():
            qml.RX(0.7, 0)
            qml.Snapshot("exact", measurement=qml.probs(wires=0), shots=None)
            return qml.sample(wires=[0])

        snapshots = qml.snapshots(circuit)()
        assert np.allclose(snapshots["exact"], [np.cos(0.35) ** 2, np.sin(0.35) ** 2], **tol)
        assert snapshots["execution_results"].shape == (50, 1)

    def test_state_snapshot_is_rejected(self, runner):
        @qml.qnode(_device(runner))
        def circuit():
            qml.Snapshot("state")
            return qml.expval(qml.Z(0))

        with pytest.raises(DeviceError, match="cannot record the state"):
            qml.snapshots(circuit)()

    @pytest.mark.parametrize(
        "measurement",
        [
            qml.state,
            lambda: qml.density_matrix(wires=[0]),
            lambda: qml.purity(wires=[0]),
            lambda: qml.vn_entropy(wires=[0]),
        ],
    )
    def test_state_measurements_are_rejected(self, runner, measurement):
        @qml.qnode(_device(runner))
        def circuit():
            qml.Hadamard(0)
            return measurement()

        with pytest.raises(DeviceError, match="needs the quantum state"):
            circuit()

    def test_probs_without_wires_on_a_wireless_device(self, runner, tol):
        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.probs()

        got = qml.QNode(circuit, _device(runner, wires=None))()
        assert np.allclose(got, [0.5, 0, 0, 0.5], **tol)

    @pytest.mark.parametrize("wires", [[2, 0, 1], ["b", "c", "a"]])
    def test_probs_without_wires_follow_default_qubit_order(self, runner, wires, tol):
        def circuit():
            qml.RY(0.4, wires[0])
            qml.RY(1.3, wires[1])
            qml.X(wires[2])
            return qml.probs()

        got = qml.QNode(circuit, _device(runner, wires=None))()
        want = qml.QNode(circuit, qml.device("default.qubit"))()
        assert np.allclose(got, want, **tol)

    def test_multi_wire_identity(self, runner, tol):
        @qml.qnode(_device(runner))
        def circuit():
            qml.RX(0.4, 0)
            qml.Identity(wires=[0, 1])
            return qml.expval(qml.Z(0))

        assert np.isclose(circuit(), np.cos(0.4), **tol)

    def test_identity_gates_are_kept(self, runner):
        @qml.qnode(_device(runner))
        def circuit():
            qml.X(0)
            qml.Identity(0)
            return qml.expval(qml.Z(0))

        circuit()
        ((_, (payload,), _),) = runner.jobs
        # The device's own padding identities are dropped, the circuit's kept.
        assert payload.circuit.count("id q[0];") == 1
        assert "id q[1];" not in payload.circuit

    def test_non_hermitian_observable(self, runner):
        @qml.qnode(_device(runner))
        def circuit():
            qml.RX(0.3, 0)
            return qml.expval(qml.X(0) @ qml.Y(0))

        with pytest.raises(DeviceError, match="not Hermitian"):
            circuit()

    def test_variance_term_limit(self, runner, monkeypatch):
        monkeypatch.setattr(divi_backend, "MAX_VARIANCE_TERMS", 2)

        @qml.qnode(_device(runner))
        def circuit():
            return qml.var(qml.Z(0) + qml.X(1) + qml.Y(2))

        with pytest.raises(DeviceError, match="analytic variances"):
            circuit()

    def test_too_many_probability_wires(self, runner):
        n = divi_backend.MAX_ANALYTIC_PROBABILITY_WIRES + 1

        @qml.qnode(_device(runner, wires=n))
        def circuit():
            return qml.probs()

        with pytest.raises(DeviceError, match="finite shots"):
            circuit()

    def test_sampling_only_backend_needs_shots(self):
        @qml.qnode(_device(SpyMaestro(force_sampling=True)))
        def circuit():
            return qml.expval(qml.Z(0))

        with pytest.raises(DeviceError, match="needs finite shots"):
            circuit()


# ---------------------------------------------------------------------------
# Finite shots
# ---------------------------------------------------------------------------


class TestShots:
    def test_samples_are_in_wire_order(self, runner):
        @qml.qnode(_device(runner), shots=20)
        def circuit():
            qml.X(0)
            qml.X(2)
            return qml.sample(), qml.counts(wires=[1, 0])

        samples, counts = circuit()
        assert samples.shape == (20, 3)
        assert np.all(samples == [1, 0, 1])
        assert counts == {"01": 20}

    def test_statistics(self, runner):
        def circuit(x, with_sample):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            qml.Hadamard(2)
            measurements = [
                qml.expval(qml.Z(0)),
                qml.probs(wires=[1]),
                qml.expval(qml.X(0) @ qml.Y(1) + qml.Z(1)),
                qml.var(qml.Z(1)),
            ]
            if with_sample:
                measurements.append(qml.sample(qml.X(2)))
            return measurements

        got = qml.set_shots(qml.QNode(circuit, _device(runner)), shots=20_000)(0.6, True)
        exact = qml.QNode(circuit, qml.device("default.qubit", wires=3))(0.6, False)
        for g, e in zip(got[:4], exact):
            assert np.allclose(g, e, atol=0.03)
        assert np.all(got[4] == 1)

    def test_pauli_words_are_exact(self, runner):
        # As on the local engine, Pauli-word expectation values ignore shots.
        @qml.qnode(_device(runner), shots=10)
        def circuit():
            qml.RX(0.6, 0)
            return qml.expval(qml.Z(0))

        assert np.isclose(circuit(), np.cos(0.6))
        assert "ham_ops" in runner.jobs[0][2]

    def test_shot_vector(self, runner):
        @qml.qnode(_device(runner), shots=[(100, 2), 300])
        def circuit():
            qml.RX(0.6, 0)
            return qml.probs(wires=0), qml.sample(wires=[0])

        result = circuit()
        assert len(result) == 3
        assert [r[1].shape for r in result] == [(100, 1), (100, 1), (300, 1)]
        # One sampling job: one circuit per shot copy, with their shots.
        ((kind, payloads, kwargs),) = runner.jobs
        assert len(payloads) == 3
        assert kwargs["shot_groups"] == [[0, 2, 100], [2, 3, 300]]

    def test_non_commuting_measurements_share_one_job(self, runner):
        @qml.qnode(_device(runner), shots=1000)
        def circuit():
            qml.Hadamard(0)
            return qml.sample(qml.X(0)), qml.sample(qml.Z(0))

        x, z = circuit()
        assert np.all(x == 1) and set(np.unique(z)) <= {-1, 1}
        assert len(runner.jobs) == 1 and len(runner.jobs[0][1]) == 2

    def test_mid_circuit_measurements(self, runner):
        @qml.qnode(_device(runner), shots=500)
        def circuit():
            qml.X(0)
            m = qml.measure(0, reset=True)
            qml.cond(m, qml.X)(1)
            return qml.sample(m), qml.sample(wires=[0, 1])

        mcm, samples = circuit()
        assert np.all(mcm == 1) and np.all(samples == [0, 1])

    def test_sampling_only_backend(self):
        @qml.qnode(_device(SpyMaestro(force_sampling=True)), shots=20_000)
        def circuit():
            qml.RX(0.6, 0)
            return qml.expval(qml.Z(0))

        assert np.isclose(circuit(), np.cos(0.6), atol=0.03)

    def test_arithmetic_of_mid_circuit_measurements(self, runner):
        def circuit():
            qml.RY(1.1, 0)
            qml.RX(0.7, 1)
            m0, m1 = qml.measure(0), qml.measure(1)
            return (
                qml.expval(m0 * m1),
                qml.expval(m0 - m1),
                qml.expval(2 * m0),
                qml.var(3 * m0),
                qml.probs(op=[m0, m1]),
            )

        got = qml.set_shots(qml.QNode(circuit, _device(runner, wires=2)), shots=50_000)()
        exact = qml.QNode(circuit, qml.device("default.qubit", wires=4))()
        for g, e in zip(got, exact):
            assert np.allclose(g, e, atol=0.03)

    def test_observables_without_a_pauli_form(self, runner):
        A = np.array([[1, 0.3 - 0.2j], [0.3 + 0.2j, -0.5]])

        def circuit():
            qml.RX(0.7, 0)
            qml.RY(-0.4, 1)
            return (
                qml.expval(qml.Hermitian(A, 0)),
                qml.var(qml.Hermitian(A, 0)),
                qml.expval(qml.Projector(np.array([1, 1]) / np.sqrt(2), 1)),
                qml.expval(qml.X(1) @ qml.Hermitian(A, 0)),
                qml.probs(op=qml.Hermitian(A, 0)),
            )

        got = qml.set_shots(qml.QNode(circuit, _device(runner)), shots=50_000)()
        exact = qml.QNode(circuit, qml.device("default.qubit", wires=3))()
        for g, e in zip(got, exact):
            assert np.allclose(g, e, atol=0.03)

    def test_identity_observables(self, runner):
        @qml.qnode(_device(runner), shots=[(30, 2), 40])
        def circuit():
            qml.Hadamard(0)
            return (
                qml.var(qml.I(0)),
                qml.sample(qml.I(0)),
                qml.counts(qml.I(0)),
                qml.expval(2 * qml.I(0) + qml.I(1)),
            )

        result = circuit()
        assert len(result) == 3
        for (var, samples, counts, expval), shots in zip(result, [30, 30, 40]):
            assert var == 0.0 and expval == 3.0
            assert samples.shape == (shots,) and np.all(samples == 1)
            assert counts == {1.0: shots}

    def test_variance_of_a_sum(self, runner):
        @qml.qnode(_device(runner), shots=50_000)
        def circuit():
            qml.RX(0.6, 0)
            qml.RY(0.4, 1)
            return qml.var(0.5 * qml.Z(0) + 0.2 * qml.X(1))

        exact = 0.25 * np.sin(0.6) ** 2 + 0.04 * np.cos(0.4) ** 2
        assert np.isclose(circuit(), exact, atol=0.01)

    def test_wireless_samples_after_mid_circuit_measurements(self, runner):
        @qml.qnode(_device(runner, wires=None), shots=200)
        def circuit():
            qml.X(0)
            qml.measure(0)
            qml.X(1)
            return qml.sample(), qml.counts()

        samples, counts = circuit()
        assert samples.shape == (200, 2) and np.all(samples == [1, 1])
        assert counts == {"11": 200}


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


def _cost(dev, **kwargs):
    @qml.qnode(dev, **kwargs)
    def circuit(x):
        _ansatz(x)
        qml.U3(x[0], x[1], x[2], 1)
        return qml.expval(H), qml.expval(qml.Z(2) @ qml.X(0))

    return circuit


def _jacobian(qnode, x):
    return qml.jacobian(lambda v: qml.math.stack(qnode(v)))(x)


class TestDerivatives:
    @pytest.mark.parametrize("device_vjp", [False, True])
    def test_adjoint_matches_backprop(self, runner, device_vjp, tol):
        got = _jacobian(_cost(_device(runner), diff_method="adjoint", device_vjp=device_vjp), X0)
        want = _jacobian(_cost(qml.device("default.qubit", wires=3), diff_method="backprop"), X0)
        assert np.allclose(got, want, **tol)

    def test_jacobian_is_one_job(self, runner):
        _jacobian(_cost(_device(runner), diff_method="adjoint"), X0)
        assert [kind for kind, *_ in runner.jobs] == ["vjp"]

    def test_best_is_adjoint(self, runner):
        assert qml.workflow.get_best_diff_method(_cost(_device(runner)))(X0) == "adjoint"

    def test_trainable_subset(self, runner, tol):
        def circuit(x, y):
            qml.RX(x, 0)
            qml.RY(y, 0)
            qml.RZ(x, 1)
            qml.CNOT([0, 1])
            return qml.expval(qml.Y(0) @ qml.X(1))

        x, y = pnp.array(0.4, requires_grad=True), pnp.array(-0.9, requires_grad=False)
        got = qml.grad(qml.QNode(circuit, _device(runner), diff_method="adjoint"))(x, y)
        want = qml.grad(qml.QNode(circuit, qml.device("default.qubit"), diff_method="backprop"))(x, y)
        assert np.allclose(got, want, **tol)

    def test_vjp_batches_tapes_of_different_widths(self, runner, tol):
        dev = _device(runner, wires=None)
        tapes = [
            QuantumScript(
                [qml.RX(0.3, 0), qml.RY(0.5, 1)],
                [qml.expval(qml.Z(0)), qml.expval(0.5 * qml.X(0) @ qml.Z(1) + 0.1 * qml.Y(1))],
                trainable_params=[0, 1],
            ),
            QuantumScript(
                [qml.RX(0.7, 0), qml.CNOT([0, 1])],
                [qml.expval(qml.X(0) @ qml.Z(1))],
                trainable_params=[0],
            ),
            QuantumScript([qml.RY(-0.2, 0)], [qml.expval(qml.X(0))], trainable_params=[0]),
        ]
        dys = [(1.0, -0.5), 2.0, 0.3]
        values, vjps = dev.execute_and_compute_vjp(tapes, dys)
        ref = qml.device("default.qubit")
        want_values, want_vjps = ref.execute_and_compute_vjp(
            tapes, dys, qml.devices.ExecutionConfig(gradient_method="adjoint")
        )
        for g, w in zip(values, want_values):
            assert np.allclose(g, w, **tol)
        for g, w in zip(vjps, want_vjps):
            assert np.allclose(g, w, **tol)
        # Widths 2, 2, 1: one VJP job per width.
        assert [kind for kind, *_ in runner.jobs] == ["vjp", "vjp"]

    def test_trainable_state_preparation(self, runner, tol):
        def circuit(x):
            qml.StatePrep(x / qml.math.linalg.norm(x), wires=[0, 1])
            qml.RY(0.3, 0)
            return qml.expval(qml.Z(0) @ qml.X(1))

        x = pnp.array([0.4, -0.2, 0.7, 0.5], requires_grad=True)
        got = qml.grad(qml.QNode(circuit, _device(runner), diff_method="adjoint"))(x)
        want = qml.grad(qml.QNode(circuit, qml.device("default.qubit"), diff_method="backprop"))(x)
        assert np.allclose(got, want, **tol)

    def test_constant_tapes_need_no_vjp(self, runner, tol):
        # A forward call through the default diff method with nothing to
        # differentiate is one expectation-value job.
        x = pnp.array(X0, requires_grad=False)
        got = _cost(_device(runner))(x)
        want = _cost(qml.device("default.qubit", wires=3))(x)
        assert np.allclose(got, want, **tol)
        assert [kind for kind, *_ in runner.jobs] == ["circuits"]

    def test_second_derivatives_use_parameter_shift(self, runner, tol):
        def circuit(x):
            qml.RX(x[0], 0)
            qml.RY(x[1], 0)
            return qml.expval(qml.Z(0))

        x = pnp.array([0.4, -0.7], requires_grad=True)
        # The device Jacobian is first order only, so max_diff=2 resolves
        # "best" to parameter-shift when the circuit runs.
        qnode = qml.QNode(circuit, _device(runner), max_diff=2)
        got = qml.jacobian(qml.grad(qnode))(x)
        want = qml.jacobian(qml.grad(qml.QNode(circuit, qml.device("default.qubit"))))(x)
        assert np.allclose(got, want, **tol)
        assert {kind for kind, *_ in runner.jobs} == {"circuits"}

    def test_missing_vjp_row_raises(self):
        with pytest.raises(DeviceError, match="no result"):
            _jacobian(_cost(_device(DropRowMaestro()), diff_method="adjoint"), X0)

    @pytest.mark.parametrize(
        "config",
        [
            MaestroConfig(simulation_type="MatrixProductState"),
            MaestroConfig(simulator_type="QiskitAer"),
        ],
    )
    def test_adjoint_needs_the_cpu_statevector(self, config):
        dev = _device(SpyMaestro(maestro_config=config))
        assert not dev.supports_derivatives()
        assert qml.workflow.get_best_diff_method(_cost(dev))(X0) == "parameter-shift"

    def test_qpu_target_has_no_adjoint(self):
        cloud = FakeQoroCloud()
        service = _offline_service(
            cloud, job_config=JobConfig(qpu_system=QPUSystem(name="fake_qpu", supports_expval=True))
        )
        assert not _device(service).supports_derivatives()


class TestGradientFallback:
    """Parameter-shift where the backend has no adjoint (any Divi release)."""

    def test_shots_use_parameter_shift(self, runner):
        qnode = _cost(_device(runner))
        qnode = qml.set_shots(qnode, shots=10)
        assert qml.workflow.get_best_diff_method(qnode)(X0) == "parameter-shift"

    def test_backend_without_vjp_uses_parameter_shift(self, tol):
        runner = NoVJPMaestro()
        assert qml.workflow.get_best_diff_method(_cost(_device(runner)))(X0) == "parameter-shift"
        got = _jacobian(_cost(_device(runner)), X0)
        want = _jacobian(_cost(qml.device("default.qubit", wires=3), diff_method="backprop"), X0)
        assert np.allclose(got, want, **tol)
        assert {kind for kind, *_ in runner.jobs} == {"circuits"}
        with pytest.raises(QuantumFunctionError):
            _cost(_device(runner), diff_method="adjoint")(X0)


# ---------------------------------------------------------------------------
# The real QoroService client, offline
# ---------------------------------------------------------------------------


def _gunzip(text):
    return gzip.decompress(base64.b64decode(text)).decode()


def _decode_observables(encoded):
    groups = []
    for group in encoded.split("|"):
        head, body = group.split(":", 1)
        n = int(head[len("@gzs") :])
        terms = []
        for sparse in _gunzip(body).split(";"):
            chars = ["I"] * n
            for pauli, index in re.findall(r"([XYZ])(\d+)", sparse):
                chars[int(index)] = pauli
            terms.append("".join(chars))
        groups.append(terms)
    return groups


class FakeQoroCloud:
    """The QoroService HTTP endpoints, computing with a local MaestroSimulator."""

    def __init__(self):
        self.sim = MaestroSimulator(maestro_config=MaestroConfig(seed=5))
        self.jobs = {}
        self.ids = itertools.count(1)

    def __call__(self, service, method, endpoint, **kwargs):
        body = kwargs.get("json")
        if endpoint == "qpusystem/":
            return mock.Mock(json=lambda: [])
        if endpoint == "simulatorcluster/":
            return mock.Mock(
                json=lambda: [
                    {"name": "qoro_maestro", "supports_expval": True},
                    {"name": "qoro_gpu", "supports_expval": True},
                ]
            )
        if endpoint == "job/init/":
            job_id = f"job-{next(self.ids)}"
            self.jobs[job_id] = {"init": body, "fragments": []}
            return mock.Mock(json=lambda: {"job_id": job_id})
        job = self.jobs[endpoint.split("/")[1]]
        if endpoint.endswith("add_circuits/"):
            job["fragments"].append(body)
            return mock.Mock(status_code=200)
        if endpoint.endswith("status/"):
            return mock.Mock(json=lambda: {"status": "COMPLETED", "run_time": 0.1})
        if "resultsV2" in endpoint:
            results = self._compute(job)
            return mock.Mock(json=lambda: {"results": results, "next": None})
        raise AssertionError(f"unexpected endpoint {endpoint}")

    def _compute(self, job):
        out, index = [], 0
        for fragment in job["fragments"]:
            groups = _decode_observables(fragment["observables"]) if "observables" in fragment else None
            ranges = fragment.get("circuit_ham_map") or [[0, 10**9]]
            if "circuits" in fragment:
                circuits = {k: _gunzip(v) for k, v in fragment["circuits"].items()}
                shots = {}
                for start, end, count in fragment.get("shot_groups") or []:
                    for k in range(start, end):
                        shots[k] = count
                for local, (label, qasm) in enumerate(circuits.items()):
                    if groups is not None:
                        group = next(g for g, (a, b) in enumerate(ranges) if a <= index < b)
                        result = self.sim.submit_circuits({label: qasm}, ham_ops=";".join(groups[group]))
                    else:
                        sim = MaestroSimulator(
                            shots=shots.get(local, fragment.get("shots", 1000)),
                            maestro_config=self.sim.maestro_config,
                        )
                        result = sim.submit_circuits({label: qasm})
                    out.append({"label": label, "results": result.results[0]["results"]})
                    index += 1
                continue
            # Parametric payloads only come from submit_vjp, which plays the
            # service's adjoint sweep with Divi's own (private) helpers.
            from divi.backends.runners._maestro import _compile_adjoint_programs, _vjp_inputs
            from divi.circuits import CircuitPayload, build_template
            from divi.circuits._qasm_template import bind_template
            from qiskit.circuit import Parameter

            template = _gunzip(fragment["circuit_template"])
            names = tuple(fragment["parameter_names"])
            (terms,) = groups
            for row in fragment["parameter_sets"]:
                payload = CircuitPayload(
                    template, tuple(Parameter(n) for n in names), ((row["label"], tuple(row["values"])),)
                )
                if "cotangent" in row:
                    ((program, matrix, offset),) = _compile_adjoint_programs([payload])
                    _, angles, weights = _vjp_inputs(
                        payload, matrix, offset, {row["label"]: np.array(row["cotangent"])}
                    )
                    raw = program.vjp(terms, weights, angles, config=self.sim.maestro_config._simulator_config())
                    out.append(
                        {
                            "label": row["label"],
                            "results": dict(zip(terms, map(float, raw["expectation_values"][0]))),
                            "metadata": {"vjp": np.asarray(raw["vjp"][0]).tolist()},
                        }
                    )
                else:
                    qasm = bind_template(build_template(template, names), row["values"])
                    result = self.sim.submit_circuits({row["label"]: qasm}, ham_ops=";".join(terms))
                    out.append({"label": row["label"], "results": result.results[0]["results"]})
        return out


def _offline_service(cloud, **kwargs):
    with mock.patch.object(QoroService, "_make_request", autospec=True, side_effect=cloud):
        service = QoroService(
            auth_token="offline",
            polling_interval=0.0,
            max_retries=5,
            use_saved_defaults=False,  # ignore any divi-config-helper defaults
            **kwargs,
        )
    service._make_request = lambda method, endpoint, **kw: cloud(service, method, endpoint, **kw)
    return service


class TestQoroService:
    @pytest.fixture
    def cloud(self):
        return FakeQoroCloud()

    def test_qoro_backend_builds_a_service(self, cloud, monkeypatch):
        monkeypatch.setattr(divi.backends, "QoroService", lambda: _offline_service(cloud))
        dev = qml.device(
            "maestro.qubit", wires=2, backend="qoro", simulation_type="MatrixProductState", max_bond_dimension=8
        )
        assert isinstance(dev.backend, QoroService)
        config = dev.backend.maestro_config
        assert config.max_bond_dimension == 8
        assert getattr(config.simulation_type, "name", config.simulation_type) == "MatrixProductState"

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.X(0))

        assert np.isclose(circuit(), 1.0)
        (job,) = cloud.jobs.values()
        assert job["init"]["job_type"] == "EXPECTATION"
        assert job["init"]["maestro_config"]["max_bond_dimension"] == 8

    def test_default_device_sends_no_maestro_options(self, cloud, monkeypatch):
        monkeypatch.setattr(divi.backends, "QoroService", lambda: _offline_service(cloud))
        dev = qml.device("maestro.qubit", wires=2, backend="qoro")
        assert dev.backend.maestro_config is None

    def test_explicit_engine_choice_is_sent(self, cloud, monkeypatch):
        # Even where it equals Maestro's default, so it beats saved defaults.
        monkeypatch.setattr(divi.backends, "QoroService", lambda: _offline_service(cloud))
        dev = qml.device("maestro.qubit", wires=2, backend="qoro", simulation_type="Statevector")
        assert "simulation_type" in dev.backend.maestro_config.model_extra

    def test_maestro_options_do_not_reach_a_qpu(self, cloud, monkeypatch):
        qpu = JobConfig(qpu_system=QPUSystem(name="fake_qpu"))
        monkeypatch.setattr(
            divi.backends, "QoroService", lambda: _offline_service(cloud, job_config=qpu)
        )
        with pytest.warns(UserWarning, match="QPU target 'fake_qpu'"):
            dev = qml.device("maestro.qubit", wires=2, backend="qoro", seed=3)
        assert dev.backend.maestro_config is None

    def test_gpu_runs_on_the_gpu_cluster(self, cloud, monkeypatch, tol):
        monkeypatch.setattr(divi.backends, "QoroService", lambda: _offline_service(cloud))
        dev = qml.device("maestro.qubit", wires=2, backend="qoro", simulator_type="Gpu")
        assert dev.backend.job_config.simulator_cluster.name == divi_backend.GPU_CLUSTER == "qoro_gpu"

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            return qml.expval(qml.Z(1))

        x = pnp.array(0.4, requires_grad=True)
        assert np.isclose(circuit(x), np.cos(0.4), **tol)
        # Maestro's adjoint needs the CPU statevector: on the GPU, parameter-shift.
        assert np.isclose(qml.grad(circuit)(x), -np.sin(0.4), **tol)
        jobs = list(cloud.jobs.values())
        assert {job["init"]["simulator_cluster"] for job in jobs} == {"qoro_gpu"}
        assert all(job["init"]["maestro_config"]["simulator_type"] == int(maestro.SimulatorType.Gpu) for job in jobs)
        assert not any("parameter_sets" in fragment for job in jobs for fragment in job["fragments"])

    def test_gpu_replaces_a_saved_qpu_target(self, cloud, monkeypatch):
        qpu = JobConfig(qpu_system=QPUSystem(name="fake_qpu"))
        monkeypatch.setattr(
            divi.backends, "QoroService", lambda: _offline_service(cloud, job_config=qpu)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dev = qml.device("maestro.qubit", wires=2, backend="qoro", simulator_type="Gpu")
        assert not [w for w in caught if "do not apply" in str(w.message)]
        assert dev.backend.job_config.qpu_system is None
        assert dev.backend.job_config.simulator_cluster.name == "qoro_gpu"
        assert dev.backend.maestro_config.simulator_type == maestro.SimulatorType.Gpu

    def test_a_passed_service_keeps_its_target(self, cloud):
        service = _offline_service(cloud)
        with pytest.warns(UserWarning, match="do not apply"):
            dev = qml.device("maestro.qubit", wires=2, backend=service, simulator_type="Gpu")
        assert dev.backend.job_config.simulator_cluster.name == "qoro_maestro"

    def test_results_match_default_qubit(self, cloud, tol):
        service = _offline_service(
            cloud,
            job_config=divi.backends.JobConfig(simulator_cluster=SimulatorCluster(name="qoro_maestro")),
        )

        def circuit(x):
            _ansatz(x)
            return qml.expval(H), qml.probs(wires=[1, 2]), qml.var(qml.X(0))

        got = qml.QNode(circuit, _device(service))(X0)
        want = qml.QNode(circuit, qml.device("default.qubit", wires=3))(X0)
        for g, w in zip(got, want):
            assert np.allclose(g, w, **tol)

    def test_counts_bit_order(self, cloud):
        service = _offline_service(cloud)

        @qml.qnode(_device(service), shots=50)
        def circuit():
            qml.X(0)
            return qml.sample(), qml.counts(wires=[0, 2])

        samples, counts = circuit()
        assert np.all(samples == [1, 0, 0]) and counts == {"10": 50}
        (job,) = cloud.jobs.values()
        assert job["init"]["job_type"] == "EXECUTE"
        assert job["fragments"][0]["shot_groups"] == [[0, 1, 50]]

    def test_adjoint_through_the_service(self, cloud, tol):
        service = _offline_service(cloud)
        got = _jacobian(_cost(_device(service), diff_method="adjoint"), X0)
        want = _jacobian(_cost(qml.device("default.qubit", wires=3), diff_method="backprop"), X0)
        assert np.allclose(got, want, **tol)
        (job,) = cloud.jobs.values()
        rows = job["fragments"][0]["parameter_sets"]
        assert len(rows) == 2 and all("cotangent" in row for row in rows)
