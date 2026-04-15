"""MaestroQubitDevice — PennyLane device backed by the Maestro simulator.

Implements the **new** ``pennylane.devices.Device`` interface (schema 3).
"""

from os import path
from typing import Union

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import decompose as _decompose
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    StateMP,
    VarianceMP,
)
from pennylane.ops.op_math import Adjoint, Conditional
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms import defer_measurements
try:
    from pennylane.transforms.core import CompilePipeline
except ImportError:
    from pennylane.transforms.core import TransformProgram as CompilePipeline
from pennylane.typing import Result, ResultBatch

import maestro
from maestro.circuits import QuantumCircuit

from pennylane_maestro.converter import (
    GATE_MAP,
    ADJOINT_MAP,
    tape_to_maestro,
    observable_to_pauli_string,
    decompose_hamiltonian_to_pauli_terms,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMULATOR_TYPE_MAP = {name: member for name, member in maestro.SimulatorType.__members__.items()}
_SIMULATION_TYPE_MAP = {name: member for name, member in maestro.SimulationType.__members__.items()}


def _resolve_enum(value, enum_map, enum_name: str):
    """Resolve a string or enum to the corresponding Maestro enum member."""
    if isinstance(value, str):
        if value not in enum_map:
            raise ValueError(
                f"Unknown {enum_name} '{value}'. "
                f"Valid options: {list(enum_map.keys())}"
            )
        return enum_map[value]
    return value


def _lsb_to_msb_statevector(state: np.ndarray, num_wires: int) -> np.ndarray:
    """Reorder a statevector from Maestro's LSB-first convention to
    PennyLane's MSB-first convention.

    Maestro qubit 0 is the *least* significant bit of the state index,
    but PennyLane expects qubit 0 to be the *most* significant bit.
    This amounts to reversing the bit pattern of every basis-state index.
    """
    n = num_wires
    size = 1 << n
    # Build a permutation: for each index i, reverse its n-bit representation
    perm = np.zeros(size, dtype=np.intp)
    for i in range(size):
        rev = 0
        val = i
        for _ in range(n):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        perm[i] = rev
    return state[perm]


def _counts_to_samples(counts: dict, num_wires: int) -> np.ndarray:
    """Expand a counts dict ``{'01': 5, '10': 3}`` into an
    ``(total_shots, num_wires)`` array of 0/1 samples."""
    samples = []
    for bitstring, count in counts.items():
        row = [int(b) for b in bitstring]
        # Pad/truncate to num_wires (should not normally be needed)
        if len(row) < num_wires:
            row = [0] * (num_wires - len(row)) + row
        for _ in range(count):
            samples.append(row)
    return np.array(samples, dtype=np.int64)


# ---------------------------------------------------------------------------
# Stopping condition for gate decomposition
# ---------------------------------------------------------------------------

def _maestro_stopping_condition(op: qml.operation.Operator) -> bool:
    """Return ``True`` for ops that the Maestro converter handles natively.

    PennyLane will decompose any op for which this returns ``False``.
    After ``defer_measurements`` has been applied, ``MidMeasureMP`` and
    ``Conditional`` nodes will already have been removed, but we keep
    them in the true-set as a safety net for when the stopping condition
    is evaluated on pre-deferred tapes.
    """
    if isinstance(op, (MidMeasureMP, Conditional)):
        return True
    if isinstance(op, Adjoint):
        base_name = op.base.name
        return base_name in GATE_MAP or base_name in ADJOINT_MAP
    return op.name in GATE_MAP


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@simulator_tracking
@single_tape_support
class MaestroQubitDevice(Device):
    """PennyLane device for the Maestro quantum simulator by Qoro Quantum.

    Args:
        wires (int or Iterable): Number of wires, or explicit wire labels.
        shots (int or None): Default shot count.  ``None`` → analytic mode.
        simulator_type (str): Backend engine. Options:

            - ``"QCSim"`` — Qoro's optimised CPU simulator (default)
            - ``"Gpu"`` — CUDA-accelerated GPU simulator
            - ``"CompositeQCSim"`` — Multi-node distributed CPU

        simulation_type (str): Simulation algorithm. Options:

            - ``"Statevector"`` — Full statevector (default)
            - ``"MatrixProductState"`` — MPS / tensor-train
            - ``"Stabilizer"`` — Clifford-only stabilizer
            - ``"TensorNetwork"`` — General tensor network
            - ``"PauliPropagator"`` — Pauli propagation
            - ``"ExtendedStabilizer"`` — Extended stabilizer

        max_bond_dimension (int or None): MPS truncation (default: None).
        singular_value_threshold (float or None): MPS SVD cutoff.
        use_double_precision (bool): Use FP64 on GPU (default: False).

    Usage examples::

        # CPU statevector (default)
        dev = qml.device("maestro.qubit", wires=20)

        # GPU statevector
        dev = qml.device("maestro.qubit", wires=20, simulator_type="Gpu")

        # GPU with double precision
        dev = qml.device("maestro.qubit", wires=20,
                         simulator_type="Gpu", use_double_precision=True)

        # MPS for large qubit counts
        dev = qml.device("maestro.qubit", wires=100,
                         simulation_type="MatrixProductState",
                         max_bond_dimension=256)

        # Stabilizer for Clifford circuits
        dev = qml.device("maestro.qubit", wires=1000,
                         simulation_type="Stabilizer")
    """

    name = "maestro.qubit"
    config_filepath = path.join(path.dirname(__file__), "config.toml")

    def __init__(
        self,
        wires=None,
        shots=None,
        simulator_type: str = "QCSim",
        simulation_type: str = "Statevector",
        max_bond_dimension=None,
        singular_value_threshold=None,
        use_double_precision: bool = False,
    ):
        super().__init__(wires=wires, shots=shots)
        self._simulator_type = _resolve_enum(
            simulator_type, _SIMULATOR_TYPE_MAP, "SimulatorType"
        )
        self._simulation_type = _resolve_enum(
            simulation_type, _SIMULATION_TYPE_MAP, "SimulationType"
        )
        self._max_bond_dimension = max_bond_dimension
        self._singular_value_threshold = singular_value_threshold
        self._use_double_precision = use_double_precision

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> CompilePipeline:
        """Return the preprocessing pipeline for this device.

        Mid-circuit measurement (MCM) strategy
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        MCMs are handled by applying
        :func:`~pennylane.transforms.defer_measurements` before gate
        decomposition.  This transform:

        * Replaces every :class:`~pennylane.measurements.MidMeasureMP` with
          an ancilla qubit and a CNOT gate (so the ancilla holds the MCM
          outcome).
        * Replaces every :class:`~pennylane.ops.op_math.Conditional`
          (classically conditioned gate) with a controlled gate driven by
          the corresponding ancilla qubit.
        * Handles ``reset=True`` MCMs by adding an additional controlled-X
          on the original wire (using the ancilla as control), resetting the
          wire to |0> while the ancilla retains the measurement value.

        The resulting tape contains only standard gates that are already in
        Maestro's native gate set, so no Maestro-side API changes are
        required.  For circuits without any MCMs the transform is a no-op,
        leaving existing behaviour unchanged.
        """
        config = execution_config or ExecutionConfig()
        mcm_method = (
            config.mcm_config.mcm_method
            if config.mcm_config is not None
            else None
        )

        from pennylane.devices.preprocess import validate_device_wires

        program = CompilePipeline()

        # ── Wire validation ──────────────────────────────────────────────
        # Expand wildcard measurements (qml.sample(), qml.counts() with no
        # args) to the full device wire set *before* defer_measurements runs.
        # Without this a 2-wire device that only has gates on wire 0 would
        # return 1-column samples instead of 2-column samples.
        program.add_transform(validate_device_wires, self.wires, name=self.name)

        # ── MCM handling ────────────────────────────────────────────────
        # defer_measurements is idempotent on circuits without MCMs, so we
        # apply it unconditionally when the user has requested (or not
        # yet specified) an MCM strategy.
        if mcm_method in {"deferred", None}:
            program.add_transform(
                defer_measurements,
                allow_postselect=False,
            )

        # ── Gate decomposition ──────────────────────────────────────────
        # After deferral, the tape contains only standard gates.
        # Decompose any gate not natively supported by the Maestro converter.
        program.add_transform(
            _decompose,
            stopping_condition=_maestro_stopping_condition,
            name=self.name,
        )

        return program

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Union[Result, ResultBatch]:
        results = []
        for tape in circuits:
            results.append(self._execute_single(tape))
        return tuple(results)

    # ------------------------------------------------------------------
    # Single tape execution
    # ------------------------------------------------------------------

    def _execute_single(self, tape: QuantumScript) -> Result:
        """Execute a single (already preprocessed) tape."""
        # Map wires to consecutive 0-indexed integers
        tape = tape.map_to_standard_wires()
        num_wires = len(tape.wires) if len(tape.wires) > 0 else 1

        is_analytic = not tape.shots

        if is_analytic:
            return self._execute_analytic(tape, num_wires)

        # Finite shots — may have a shot vector
        shot_results = []
        for shot_copy in tape.shots:
            shot_results.append(
                self._execute_finite_shots(tape, num_wires, shot_copy)
            )

        if not tape.shots.has_partitioned_shots:
            return shot_results[0]
        return tuple(shot_results)

    # ------------------------------------------------------------------
    # Analytic execution  (shots=None)
    # ------------------------------------------------------------------

    def _execute_analytic(self, tape: QuantumScript, num_wires: int) -> Result:
        """Compute results from exact simulation.

        Fast path: if every measurement is an expectation value of a Pauli
        observable, delegate to Maestro's ``estimate()`` which computes
        exact expectation values without materialising the full 2^n
        statevector.  This is dramatically faster for large qubit counts.

        Slow path: fall back to full statevector extraction for anything
        else (variance, probabilities, state, Hermitian observables, etc.).
        """
        # ── Fast path: all-Pauli expval → use estimate() ──
        all_pauli_expval = all(
            isinstance(mp, ExpectationMP)
            and observable_to_pauli_string(mp.obs, num_wires) is not None
            for mp in tape.measurements
        )
        if all_pauli_expval:
            return self._execute_estimate(tape, num_wires)

        # ── Fast path: Hamiltonian / Sum expval → batched estimate() ──
        all_hamiltonian_expval = all(
            isinstance(mp, ExpectationMP)
            and decompose_hamiltonian_to_pauli_terms(mp.obs, num_wires) is not None
            for mp in tape.measurements
        )
        if all_hamiltonian_expval:
            return self._execute_hamiltonian(tape, num_wires)

        # ── Slow path: full statevector ──
        qc = tape_to_maestro(tape, num_wires)

        kwargs = dict(
            simulator_type=self._simulator_type,
            simulation_type=self._simulation_type,
            use_double_precision=self._use_double_precision,
        )
        if self._max_bond_dimension is not None:
            kwargs["max_bond_dimension"] = self._max_bond_dimension
        if self._singular_value_threshold is not None:
            kwargs["singular_value_threshold"] = self._singular_value_threshold

        amplitudes = qc.get_statevector(**kwargs)
        state = np.array(amplitudes, dtype=np.complex128)
        state = _lsb_to_msb_statevector(state, num_wires)

        results = tuple(
            mp.process_state(state, tape.wires) for mp in tape.measurements
        )
        if len(tape.measurements) == 1:
            return results[0]
        return results

    # ------------------------------------------------------------------
    # Finite-shots execution
    # ------------------------------------------------------------------

    def _execute_finite_shots(
        self, tape: QuantumScript, num_wires: int, shots: int
    ) -> Result:
        """Execute with a finite number of shots."""
        # Check if ALL measurements can use the estimate fast-path
        # (Pauli expvals only — no samples, counts, probs, etc.)
        all_pauli_expval = all(
            isinstance(mp, ExpectationMP)
            and observable_to_pauli_string(mp.obs, num_wires) is not None
            for mp in tape.measurements
        )

        if all_pauli_expval:
            return self._execute_estimate(tape, num_wires)

        # General path: sample from counts
        qc = tape_to_maestro(tape, num_wires)
        qc.measure_all()

        kwargs = dict(
            simulator_type=self._simulator_type,
            simulation_type=self._simulation_type,
            shots=shots,
            use_double_precision=self._use_double_precision,
        )
        if self._max_bond_dimension is not None:
            kwargs["max_bond_dimension"] = self._max_bond_dimension
        if self._singular_value_threshold is not None:
            kwargs["singular_value_threshold"] = self._singular_value_threshold

        raw = qc.execute(**kwargs)
        counts = raw["counts"]

        samples = _counts_to_samples(counts, num_wires)

        results = tuple(
            mp.process_samples(samples, tape.wires) for mp in tape.measurements
        )
        if len(tape.measurements) == 1:
            return results[0]
        return results

    # ------------------------------------------------------------------
    # Fast Pauli-expectation path via Maestro estimate()
    # ------------------------------------------------------------------

    def _execute_estimate(
        self, tape: QuantumScript, num_wires: int
    ) -> Result:
        """Use Maestro's estimate() for pure-Pauli expval measurements."""
        qc = tape_to_maestro(tape, num_wires)

        pauli_strings = []
        for mp in tape.measurements:
            ps = observable_to_pauli_string(mp.obs, num_wires)
            pauli_strings.append(ps)

        kwargs = dict(
            simulator_type=self._simulator_type,
            simulation_type=self._simulation_type,
            use_double_precision=self._use_double_precision,
        )
        if self._max_bond_dimension is not None:
            kwargs["max_bond_dimension"] = self._max_bond_dimension
        if self._singular_value_threshold is not None:
            kwargs["singular_value_threshold"] = self._singular_value_threshold

        raw = qc.estimate(observables=pauli_strings, **kwargs)
        exp_vals = raw["expectation_values"]

        # Handle SProd (scalar * Pauli) — multiply by the scalar
        results = []
        for mp, ev in zip(tape.measurements, exp_vals):
            if isinstance(mp.obs, qml.ops.SProd):
                ev = float(mp.obs.scalar) * ev
            results.append(np.float64(ev))

        if len(results) == 1:
            return results[0]
        return tuple(results)

    # ------------------------------------------------------------------
    # Batched Hamiltonian path via Maestro estimate()
    # ------------------------------------------------------------------

    def _execute_hamiltonian(
        self, tape: QuantumScript, num_wires: int
    ) -> Result:
        """Evaluate Hamiltonian/Sum expvals via a single batched estimate().

        All Pauli terms across all Hamiltonian measurements are collected
        into one list, sent to Maestro's ``estimate()`` in a single C++
        call, and the weighted sum is computed in Python.
        """
        qc = tape_to_maestro(tape, num_wires)

        # Collect all Pauli terms and track which measurement owns which
        all_pauli_strings = []
        term_slices = []  # (start_idx, count, coeffs) per measurement

        for mp in tape.measurements:
            terms = decompose_hamiltonian_to_pauli_terms(mp.obs, num_wires)
            start = len(all_pauli_strings)
            coeffs = []
            for coeff, ps in terms:
                coeffs.append(coeff)
                all_pauli_strings.append(ps)
            term_slices.append((start, len(terms), coeffs))

        kwargs = dict(
            simulator_type=self._simulator_type,
            simulation_type=self._simulation_type,
            use_double_precision=self._use_double_precision,
        )
        if self._max_bond_dimension is not None:
            kwargs["max_bond_dimension"] = self._max_bond_dimension
        if self._singular_value_threshold is not None:
            kwargs["singular_value_threshold"] = self._singular_value_threshold

        raw = qc.estimate(observables=all_pauli_strings, **kwargs)
        all_exp_vals = raw["expectation_values"]

        # Reconstruct each Hamiltonian expval as Σ cᵢ⟨Pᵢ⟩
        results = []
        for start, count, coeffs in term_slices:
            expval = sum(
                c * ev for c, ev in zip(coeffs, all_exp_vals[start:start + count])
            )
            results.append(np.float64(expval))

        if len(results) == 1:
            return results[0]
        return tuple(results)
